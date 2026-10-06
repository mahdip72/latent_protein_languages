import argparse
import os
import time
from typing import Any, Dict, List
import csv

import torch
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm

from data.dataset import ProteinTokenizer
from models.super_model import prepare_models
from utils.inference.de_novo import (
    _decode_denovo_structures,
)
from utils.log import get_logging
from utils.model import clean_state_dict_keys, compile_model
from utils.utils import (
    load_yaml_box,
    load_trained_run_configs,
    create_inference_result_dir,
    compute_decoded_content_length,
    check_eos_and_max_length,
    compute_amino_acid_sequence_entropy,
    resolve_sequence_entropy_filter_config,
)

def main(args: argparse.Namespace) -> None:
    infer_cfg = load_yaml_box(args.config_path)
    condition_cfg = infer_cfg.get('condition_tokens', {})
    ensure_eos_samples = bool(infer_cfg.get('ensure_num_samples_with_eos', False))
    exclude_max_length_sequences = bool(infer_cfg.get('exclude_max_length_sequences', False))

    dataloader_cfg = DataLoaderConfiguration(
        non_blocking=True,
        even_batches=False,
    )
    # Set NCCL timeout to 60 minutes for large-scale generation
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_cfg,
        kwargs_handlers=[process_group_kwargs],
    )

    result_dir = create_inference_result_dir(
        accelerator, 
        infer_cfg.output_base_dir, 
        args.config_path
    )
    trained_dir = infer_cfg.trained_model_dir
    
    # Extract model name from trained_model_dir (the timestamp folder name)
    model_name = os.path.basename(os.path.normpath(trained_dir))
    
    configs = load_trained_run_configs(trained_dir, infer_cfg.config_filename)
    
    # Get data_type from the training config
    data_type = getattr(configs.model, 'data_type', 'amino_acid')
    entropy_filter_cfg = resolve_sequence_entropy_filter_config(infer_cfg, data_type)
    entropy_filter_enabled = bool(entropy_filter_cfg["enabled"])
    entropy_min_threshold = entropy_filter_cfg["min_entropy"]

    logger = get_logging(result_dir, configs)
    logger.info("Starting de novo sequence generation")
    logger.info(f"Trained model directory: {trained_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Data type: {data_type}")
    logger.info(f"Number of samples to generate: {infer_cfg.num_samples}")
    if ensure_eos_samples:
        logger.info("Termination enforcement enabled: will keep generating until requested samples include EOS tokens.")
    if exclude_max_length_sequences:
        logger.info("Will discard sequences that reach the generation max length (even if they contain EOS).")
    if entropy_filter_enabled:
        logger.info(f"Sequence entropy filter enabled: keeping amino-acid sequences with entropy >= {entropy_min_threshold:.4f}.")
    elif entropy_filter_cfg["requested"]:
        logger.warning(
            "sequence_entropy_filter is enabled but the loaded model data_type is %s. "
            "On-the-fly entropy filtering is currently supported only for direct amino-acid generation; skipping it.",
            data_type,
        )
    if data_type == "pll":
        seq_column = "pll_sequence"
    elif data_type == "structure":
        seq_column = "structure_sequence"
    else:
        seq_column = "amino_acid_sequence"
    
    # Initialize tokenizer with training condition config to match checkpoint vocabulary
    # The inference condition config controls generation behavior, not tokenizer vocabulary
    training_condition_config = getattr(configs.model, 'condition_tokens', None)

    tokenizer = ProteinTokenizer(
        data_type,
        condition_config=training_condition_config,
        max_len=configs.model.max_len,
        vocab_path=(os.path.join(trained_dir, 'tokenizer_vocab.yaml')
                    if os.path.isfile(os.path.join(trained_dir, 'tokenizer_vocab.yaml')) else None),
    )
    logger.info(f"Tokenizer vocab size: {tokenizer.tokenizer_vocab_size}")
    logger.info(f"BOS token ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
    logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
    if tokenizer.conditioning_enabled:
        logger.info("Inference conditioning enabled:")
        for cond in tokenizer.conditions:
            logger.info(f"  - {cond.name} (prob=1.0 forced)")

    # Load checkpoint before model construction so older amino-acid checkpoints
    # with no X token can keep their original embedding size.
    checkpoint_path = os.path.join(trained_dir, infer_cfg.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = clean_state_dict_keys(state_dict)

    tokenizer_vocab_size = tokenizer.tokenizer_vocab_size
    token_emb_key = next(
        (
            key for key in (
                'transformer.token_emb.emb.weight',
                'transformer.net.token_emb.emb.weight',
            )
            if key in state_dict
        ),
        None,
    )
    if token_emb_key is not None:
        checkpoint_vocab_size = int(state_dict[token_emb_key].shape[0])
        if checkpoint_vocab_size != tokenizer_vocab_size:
            if data_type == 'amino_acid' and checkpoint_vocab_size == tokenizer_vocab_size - 1:
                logger.warning(
                    "Using checkpoint vocab size %d instead of tokenizer vocab size %d "
                    "for backward-compatible amino-acid inference.",
                    checkpoint_vocab_size,
                    tokenizer_vocab_size,
                )
                tokenizer_vocab_size = checkpoint_vocab_size
            else:
                raise ValueError(
                    f"Checkpoint vocab size {checkpoint_vocab_size} does not match "
                    f"tokenizer vocab size {tokenizer_vocab_size}"
                )

    # Build inference model using prepare_models with inference=True
    model = prepare_models(
        configs, 
        logger, 
        inference=True,
        tokenizer_vocab_size=tokenizer_vocab_size,
        pad_token_id=tokenizer.pad_token_id
    )
    model.eval()

    # Map keys from training model to inference model
    # Training model: transformer.xxx -> Inference model: transformer.net.xxx
    # (because inference mode wraps transformer with AutoregressiveWrapper)
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('transformer.'):
            # Training: transformer.xxx -> Inference: transformer.net.xxx
            new_key = key.replace('transformer.', 'transformer.net.', 1)
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value
    
    load_log = model.load_state_dict(mapped_state_dict, strict=False)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint loading log - Missing keys: {len(load_log.missing_keys)}, Unexpected keys: {len(load_log.unexpected_keys)}")
    if load_log.missing_keys:
        logger.info(f"Missing keys: {load_log.missing_keys[:10]}...")  # Show first 10
    if load_log.unexpected_keys:
        logger.info(f"Unexpected keys: {load_log.unexpected_keys[:10]}...")  # Show first 10

    # Compile model if enabled
    compile_cfg = infer_cfg.get('compile_model', {})
    if compile_cfg.get('enabled', False):
        model = compile_model(model, mode=compile_cfg.get('mode', None))
        logger.info("Compiled model for inference.")

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Get sampling parameters
    sampling_cfg = infer_cfg.get('sampling', {})
    temperature = sampling_cfg.get('temperature', 1.0)
    filter_logits_fn = sampling_cfg.get('filter_logits_fn', None)
    filter_kwargs = sampling_cfg.get('filter_kwargs', {})
    
    logger.info(f"Sampling parameters - Temperature: {temperature}, Filter: {filter_logits_fn}, Filter kwargs: {filter_kwargs}")

    # Calculate batches
    num_samples = int(infer_cfg.num_samples)
    batch_size = int(infer_cfg.batch_size)
    max_gen_len = int(infer_cfg.max_generation_length)
    
    # Distribute samples across processes
    num_processes = accelerator.num_processes
    samples_per_process = (num_samples + num_processes - 1) // num_processes
    
    # Adjust for this process
    process_index = accelerator.process_index
    start_sample = process_index * samples_per_process
    end_sample = min(start_sample + samples_per_process, num_samples)
    local_num_samples = end_sample - start_sample
    
    if local_num_samples <= 0:
        local_num_samples = 0
        num_batches = 0
    else:
        num_batches = (local_num_samples + batch_size - 1) // batch_size
    
    # Calculate total batches across all processes
    total_batches = (num_samples + batch_size - 1) // batch_size
    if accelerator.is_main_process:
        logger.info(f"Generating {num_samples} samples in {total_batches} batches (batch_size={batch_size}, max_len={max_gen_len})")

    progress_bar = tqdm(
        total=num_batches,
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
        leave=True,
    )
    progress_bar.set_description("De novo generation")

    records: List[Dict[str, Any]] = []
    merge_joiner = infer_cfg.get('merge_joiner', '')
    total_tokens_generated = 0  # Track actual tokens (excluding post-EOS) for real TPS
    total_tokens_processed = 0  # Track all tokens processed by model for peak TPS
    entropy_rejected_samples = 0
    
    samples_generated_for_process = 0

    # Start timing generation
    accelerator.wait_for_everyone()
    generation_start_time = time.time()

    # Build optional conditioning prefix tokens for prompts
    conditioning_prefix_tokens: List[str] = []
    if tokenizer.conditioning_enabled:
        # Order matches tokenizer conditions list
        if condition_cfg.get('c2n', {}).get('enabled', False):
            conditioning_prefix_tokens.append('<C2N>')
        if condition_cfg.get('length', {}).get('enabled', False):
            fixed_len = condition_cfg.get('length', {}).get('fixed_value', None)
            if fixed_len is None:
                raise ValueError("condition_tokens.length.enabled is True but fixed_value is not set in inference config")
            fixed_len = int(fixed_len)
            max_allowed = max(1, configs.model.max_len - 4)
            if fixed_len > max_allowed:
                raise ValueError(f"condition_tokens.length.fixed_value={fixed_len} exceeds allowed max {max_allowed} for tokenizer length tokens")
            conditioning_prefix_tokens.append(f'<LEN_{int(fixed_len)}>')
        # Always include BOP when conditioning mode is on
        conditioning_prefix_tokens.append('<BOP>')

    # Convert prefix tokens to IDs
    conditioning_prefix_ids: List[int] = [
        tokenizer.token_to_id.get(tok, tokenizer.unk_token_id) for tok in conditioning_prefix_tokens
    ]

    # Adjust generation length to preserve content budget when prefix is present
    generation_seq_len = max_gen_len + len(conditioning_prefix_ids)
    # Clamp to model max_len to avoid overflow
    generation_seq_len = min(generation_seq_len, configs.model.max_len)

    while samples_generated_for_process < local_num_samples:
        # Determine batch size for this iteration
        remaining_samples = local_num_samples - samples_generated_for_process
        current_batch_size = min(batch_size, remaining_samples)
        
        if current_batch_size <= 0:
            break
        
        with torch.inference_mode():
            # Create prompts with just BOS token
            # Build prompts with BOS (+ optional conditioning prefix + BOP)
            prompt_tokens = [tokenizer.bos_token_id] + conditioning_prefix_ids
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=accelerator.device)
            prompts = prompt_tensor.unsqueeze(0).repeat(current_batch_size, 1)
            
            # Generate sequences using the model's generate method
            generated = model.generate(
                prompts=prompts,
                seq_len=generation_seq_len,
                eos_token=tokenizer.eos_token_id,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
            )
            
            # Track tokens processed by model (batch_size × generation_steps)
            # This counts ALL tokens including those generated after EOS
            total_tokens_processed += generated.shape[0] * generated.shape[1]
            
            # Process generated sequences
            generated_cpu = generated.detach().cpu()
            
            for idx_in_batch in range(current_batch_size):
                seq_tokens = generated_cpu[idx_in_batch]
                
                # Check for EOS token and max length conditions
                has_eos, first_eos_pos, hit_max_len = check_eos_and_max_length(
                    seq_tokens, tokenizer.eos_token_id, generation_seq_len
                )
                if exclude_max_length_sequences and hit_max_len:
                    continue
                if has_eos:
                    actual_length = first_eos_pos
                    seq_tokens = seq_tokens[:actual_length]
                else:
                    actual_length = len(seq_tokens)

                if ensure_eos_samples and not has_eos:
                    continue
                
                # Track total tokens generated
                total_tokens_generated += actual_length
                
                # Decode sequence
                decoded_sequence = tokenizer.decode(
                    seq_tokens, 
                    skip_special_tokens=True,
                    skip_condition_tokens=True,
                    joiner=merge_joiner
                )

                # For CSV reporting, we want length to reflect the *content* length (protein tokens),
                # not the raw generated token length which can include conditioning / special tokens.
                decoded_length = compute_decoded_content_length(decoded_sequence, merge_joiner)
                sequence_entropy = None
                if entropy_filter_enabled:
                    sequence_entropy = compute_amino_acid_sequence_entropy(decoded_sequence)
                    if sequence_entropy < entropy_min_threshold:
                        entropy_rejected_samples += 1
                        continue

                sample_global_idx = start_sample + samples_generated_for_process + 1  # 1-indexed
                
                record = {
                    'model_name': model_name,
                    'sample_index': sample_global_idx,
                    'data_type': data_type,
                    'length': decoded_length,
                    'has_eos_token': has_eos,
                    seq_column: decoded_sequence,
                }
                if entropy_filter_enabled:
                    record['entropy'] = sequence_entropy
                records.append(record)
                samples_generated_for_process += 1

                if ensure_eos_samples and samples_generated_for_process >= local_num_samples:
                    break
            
        if (ensure_eos_samples or exclude_max_length_sequences or entropy_filter_enabled) and progress_bar.n >= progress_bar.total:
            progress_bar.total += 1
            progress_bar.refresh()
        progress_bar.update(1)

    progress_bar.close()

    # End timing generation
    accelerator.wait_for_everyone()
    generation_end_time = time.time()
    generation_elapsed_time = generation_end_time - generation_start_time

    # Gather token counts from all processes (small tensor, safe to gather)
    token_counts_tensor = torch.tensor(
        [total_tokens_generated, total_tokens_processed, entropy_rejected_samples],
        device=accelerator.device
    )
    gathered_token_counts = accelerator.gather(token_counts_tensor)
    # Reshape from [num_processes * 3] to [num_processes, 3]
    gathered_token_counts = gathered_token_counts.view(-1, 3)
    total_tokens_all_processes = gathered_token_counts[:, 0].sum().item()  # Real tokens
    total_processed_all_processes = gathered_token_counts[:, 1].sum().item()  # All processed
    total_entropy_rejected = gathered_token_counts[:, 2].sum().item()

    # Write records to temporary per-process files to avoid NCCL timeout on large gathers
    temp_csv_path = os.path.join(result_dir, f'_temp_rank_{process_index}.csv')
    with open(temp_csv_path, 'w', newline='') as temp_file:
        temp_writer = csv.writer(temp_file)
        for rec in records:
            row = [
                rec['model_name'],
                rec['sample_index'],
                rec['data_type'],
                rec['length'],
                rec['has_eos_token'],
                rec[seq_column],
            ]
            if entropy_filter_enabled:
                row.append(rec.get('entropy'))
            temp_writer.writerow(row)
    
    # Synchronize all processes after writing temp files
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Read and merge all temporary files
        all_records = []
        for rank in range(num_processes):
            rank_csv_path = os.path.join(result_dir, f'_temp_rank_{rank}.csv')
            if os.path.exists(rank_csv_path):
                with open(rank_csv_path, 'r', newline='') as rank_file:
                    reader = csv.reader(rank_file)
                    for row in reader:
                        all_records.append({
                            'model_name': row[0],
                            'sample_index': int(row[1]),
                            'data_type': row[2],
                            'length': int(row[3]),
                            'has_eos_token': row[4].lower() == 'true',
                            seq_column: row[5],
                        })
                        if entropy_filter_enabled and len(row) > 6:
                            all_records[-1]['entropy'] = float(row[6])
                # Remove temporary file
                os.remove(rank_csv_path)
        
        # Sort by sample_index to ensure order
        all_records.sort(key=lambda x: x['sample_index'])
        
        # Deduplicate (in case of overlapping assignments)
        seen_indices = set()
        unique_records = []
        for rec in all_records:
            if rec['sample_index'] not in seen_indices:
                seen_indices.add(rec['sample_index'])
                unique_records.append(rec)
        
        # Limit to requested number of samples
        unique_records = unique_records[:num_samples]
        
        # Log generation timing statistics
        num_generated = len(unique_records)
        samples_per_second = num_generated / generation_elapsed_time if generation_elapsed_time > 0 else 0
        
        # Real TPS: actual useful tokens (excluding post-EOS tokens)
        real_tps = total_tokens_all_processes / generation_elapsed_time if generation_elapsed_time > 0 else 0
        # Peak TPS: all tokens processed by model (including post-EOS, reflects actual compute)
        peak_tps = total_processed_all_processes / generation_elapsed_time if generation_elapsed_time > 0 else 0
        
        avg_tokens_per_sample = total_tokens_all_processes / num_generated if num_generated > 0 else 0
        efficiency = (total_tokens_all_processes / total_processed_all_processes * 100) if total_processed_all_processes > 0 else 0
        
        logger.info(f"Generation timing statistics:")
        logger.info(f"  Total samples generated: {num_generated}")
        logger.info(f"  Total tokens (real): {total_tokens_all_processes:,}")
        logger.info(f"  Total tokens (processed): {total_processed_all_processes:,}")
        logger.info(f"  Token efficiency: {efficiency:.1f}% (real/processed)")
        logger.info(f"  Average tokens per sample: {avg_tokens_per_sample:.1f}")
        logger.info(f"  Total generation time: {generation_elapsed_time:.2f} seconds")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/second")
        logger.info(f"  Real TPS: {real_tps:.2f} tokens/second (useful tokens)")
        logger.info(f"  Peak TPS: {peak_tps:.2f} tokens/second (model compute)")
        logger.info(f"  Average time per sample: {generation_elapsed_time / num_generated * 1000:.2f} ms")

        csv_filename = infer_cfg.get('output_csv_filename', 'generated_sequences.csv')
        csv_path = os.path.join(result_dir, csv_filename)

        num_with_eos = sum(1 for rec in unique_records if rec['has_eos_token'])
        logger.info(f"Sequences containing EOS token: {num_with_eos}/{len(unique_records)}")
        if entropy_filter_enabled:
            logger.info(f"Sequences rejected by entropy filter: {int(total_entropy_rejected)}")

        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ['model_name', 'sample_index', 'data_type', 'length', 'has_eos_token', seq_column]
            if entropy_filter_enabled:
                header.append('entropy')
            csv_writer.writerow(header)
            for rec in unique_records:
                row = [
                    rec['model_name'],
                    rec['sample_index'],
                    rec['data_type'],
                    rec['length'],
                    rec['has_eos_token'],
                    rec[seq_column],
                ]
                if entropy_filter_enabled:
                    row.append(rec.get('entropy'))
                csv_writer.writerow(row)

        logger.info(f"Generated {len(unique_records)} sequences")
        logger.info(f"Saved generated sequences to {csv_path}")

    accelerator.wait_for_everyone()

    csv_filename = infer_cfg.get('output_csv_filename', 'generated_sequences.csv')
    csv_path = os.path.join(result_dir, csv_filename)
    _decode_denovo_structures(
        data_type=data_type,
        infer_cfg=infer_cfg,
        accelerator=accelerator,
        logger=logger,
        result_dir=result_dir,
        csv_path=csv_path,
        seq_column=seq_column,
        merge_joiner=merge_joiner,
    )

    accelerator.free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate de novo protein sequences using a trained language model.")
    parser.add_argument(
        "--config_path",
        "-c",
        default="configs/inference_generate_de_novo_config.yaml",
        help="Path to the generation inference configuration file.",
    )
    main(parser.parse_args())
