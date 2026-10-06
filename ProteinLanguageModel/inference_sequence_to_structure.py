import argparse
import os
import time
from datetime import timedelta
from typing import Any, Dict

import pandas as pd
import torch
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ProteinTokenizer
from data.utils import select_modality_column
from models.super_model import prepare_models
from utils.inference.sequence_to_structure import (
    build_protein_context_batch,
    SequenceRowDataset,
    SequenceToStructureDataset,
    StructureIndicesDataset,
    compute_token_metrics,
    gather_sequence_values,
    postprocess_generated_batch,
    prepare_inference_state_dict,
    prepare_prompt_tokens,
    save_backbone_pdb_inference,
    sequence_row_collate,
    sequence_to_structure_collate,
    validate_condition_support,
    write_rank_predictions_csv,
    build_decode_records,
)
from utils.log import get_logging
from utils.model import (
    clean_state_dict_keys,
    compile_model,
)
from utils.utils import (
    create_inference_result_dir,
    load_trained_run_configs,
    load_yaml_box,
)


STRUCTURE_CODEBOOK_SIZE = 4096


def _cleanup_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_vqvae_decoder(
    vqvae_decode_cfg: Dict[str, Any],
    infer_cfg: Box,
    device: torch.device,
    logger,
):
    from gcp_vqvae import GCPVQVAE

    trained_model_dir = vqvae_decode_cfg.get("trained_model_dir")
    if not trained_model_dir:
        raise ValueError("vqvae_decode.trained_model_dir is required when decoding")

    mixed_precision = vqvae_decode_cfg.get("mixed_precision", infer_cfg.mixed_precision)
    deterministic = bool(vqvae_decode_cfg.get("deterministic", False))
    seed = int(vqvae_decode_cfg.get("seed", 0))

    return GCPVQVAE(
        trained_model_dir=trained_model_dir,
        checkpoint_path=vqvae_decode_cfg.get("checkpoint_path", "checkpoints/best_valid.pth"),
        config_vqvae=vqvae_decode_cfg.get("config_vqvae", "config_vqvae.yaml"),
        config_encoder=vqvae_decode_cfg.get("config_encoder", "config_gcpnet_encoder.yaml"),
        config_decoder=vqvae_decode_cfg.get("config_decoder", "config_geometric_decoder.yaml"),
        mode="decode",
        device=str(device),
        mixed_precision=mixed_precision,
        max_length=vqvae_decode_cfg.get("max_length"),
        deterministic=deterministic,
        seed=seed,
        logger=logger,
    )


def _build_pll_model(
    pll_cfg: Dict[str, Any],
    device: torch.device,
    mixed_precision: str,
):
    from pll import PLL

    config_path = pll_cfg.get("config_path")
    checkpoint_path = pll_cfg.get("checkpoint_path")
    if not config_path or not checkpoint_path:
        raise ValueError("pll.config_path and pll.checkpoint_path are required when sequence modality is pll")

    pll_device = pll_cfg.get("device")
    pll_mixed_precision = pll_cfg.get("mixed_precision", mixed_precision)
    pll_version = str(pll_cfg.get("version", "1"))

    return PLL(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        mode="encode",
        version=pll_version,
        device=pll_device or str(device),
        mixed_precision=pll_mixed_precision,
    )


def main(args: argparse.Namespace) -> None:
    infer_cfg = load_yaml_box(args.config_path)
    condition_cfg = infer_cfg.get("condition_tokens", {})

    dataloader_cfg = DataLoaderConfiguration(
        non_blocking=True,
        even_batches=False,
    )
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=600))
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_cfg,
        kwargs_handlers=[process_group_kwargs],
    )

    result_dir = create_inference_result_dir(
        accelerator,
        infer_cfg.output_base_dir,
        args.config_path,
    )
    trained_dir = infer_cfg.trained_model_dir
    model_name = os.path.basename(os.path.normpath(trained_dir))

    configs = load_trained_run_configs(trained_dir, infer_cfg.config_filename)
    data_type = getattr(configs.model, "data_type", "amino_acid")

    logger = get_logging(result_dir, configs)
    logger.info("Starting sequence-to-structure inference")
    logger.info("Trained model directory: %s", trained_dir)
    logger.info("Model name: %s", model_name)
    logger.info("Data type: %s", data_type)
    logger.info("Input CSV: %s", infer_cfg.input_csv_path)

    training_condition_config = getattr(configs.model, "condition_tokens", None)
    tokenizer = ProteinTokenizer(
        data_type,
        condition_config=training_condition_config,
        max_len=configs.model.max_len,
        vocab_path=(os.path.join(trained_dir, 'tokenizer_vocab.yaml')
                    if os.path.isfile(os.path.join(trained_dir, 'tokenizer_vocab.yaml')) else None),
    )
    validate_condition_support(tokenizer, condition_cfg)

    sequence_modality = tokenizer.sequence_to_structure_sequence_modality
    if not sequence_modality:
        raise ValueError("sequence_to_structure requires a sequence modality (use_pll) in the training config")

    input_csv_path = infer_cfg.input_csv_path
    df = pd.read_csv(input_csv_path)

    sequence_column_override = infer_cfg.get("sequence_column_name", None)
    sequence_column = select_modality_column(
        df.columns.tolist(),
        "amino_acid",
        override=sequence_column_override,
    )
    logger.info(
        "Using input sequence column: '%s' (model modality: %s)",
        sequence_column,
        sequence_modality,
    )

    total_rows = len(df)
    num_samples = int(infer_cfg.num_samples)
    if num_samples < 0 or num_samples > total_rows:
        num_samples = total_rows
    df = df.head(num_samples).reset_index(drop=True)
    logger.info("Number of samples: %d", num_samples)

    batch_size = int(infer_cfg.batch_size)
    max_generation_length = int(infer_cfg.max_generation_length)
    num_generations_per_sample = int(infer_cfg.get("num_generations_per_sample", 1))
    if num_generations_per_sample < 1:
        raise ValueError("num_generations_per_sample must be >= 1")
    if batch_size < num_generations_per_sample:
        logger.warning(
            "batch_size (%d) < num_generations_per_sample (%d); generations will span multiple batches.",
            batch_size,
            num_generations_per_sample,
        )

    id_column = infer_cfg.get("id_column_name")
    sequence_values = None
    if sequence_modality == "pll":
        pll_cfg = infer_cfg.get("pll", {})
        pll_batch_size = int(pll_cfg.get("batch_size", 32))
        pll_num_workers = int(pll_cfg.get("num_workers", 0))
        pll_model = _build_pll_model(
            pll_cfg,
            accelerator.device,
            infer_cfg.mixed_precision,
        )
        pll_dataset = SequenceRowDataset(
            df,
            sequence_column=sequence_column,
        )
        pll_loader = DataLoader(
            pll_dataset,
            batch_size=pll_batch_size,
            shuffle=False,
            num_workers=pll_num_workers,
            collate_fn=sequence_row_collate,
        )
        pll_loader = accelerator.prepare(pll_loader)

        local_pairs = []
        for batch in pll_loader:
            sequences = batch["sequence"]
            row_indices = batch["row_index"]
            if not sequences:
                continue
            records = pll_model.encode_batch(sequences)
            if len(records) != len(sequences):
                raise ValueError("PLL returned mismatched record count.")
            for row_idx, rec in zip(row_indices, records):
                local_pairs.append((int(row_idx), rec["indices_str"]))

        del pll_model
        _cleanup_memory(accelerator.device)
        sequence_values = gather_sequence_values(local_pairs, len(df), accelerator)

    dataset = SequenceToStructureDataset(
        df,
        sequence_column=sequence_column,
        num_generations_per_sample=num_generations_per_sample,
        id_column=id_column,
        sequence_values=sequence_values,
    )
    num_workers = int(infer_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sequence_to_structure_collate,
    )
    loader = accelerator.prepare(loader)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    progress_bar = tqdm(
        total=len(loader),
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
        leave=True,
    )
    progress_bar.set_description("Structure prediction")

    model = prepare_models(
        configs,
        logger,
        inference=True,
        tokenizer_vocab_size=tokenizer.tokenizer_vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.eval()

    checkpoint_path = os.path.join(trained_dir, infer_cfg.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = clean_state_dict_keys(state_dict)
    filtered_state = prepare_inference_state_dict(
        state_dict,
        model,
        tokenizer,
        checkpoint_path,
        logger,
    )

    load_log = model.load_state_dict(filtered_state, strict=False)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    logger.info(
        "Checkpoint loading log - Missing keys: %s, Unexpected keys: %s",
        len(load_log.missing_keys),
        len(load_log.unexpected_keys),
    )

    compile_cfg = infer_cfg.get("compile_model", {})
    if compile_cfg.get("enabled", False):
        model = compile_model(model, mode=compile_cfg.get("mode", None))
        logger.info("Compiled model for inference.")

    model = accelerator.prepare(model)

    sampling_cfg = infer_cfg.get("sampling", {})
    temperature = sampling_cfg.get("temperature", 1.0)
    filter_logits_fn = sampling_cfg.get("filter_logits_fn", None)
    filter_kwargs = sampling_cfg.get("filter_kwargs", {})
    return_token_probs = bool(infer_cfg.get("return_token_probs", False))
    return_token_entropy = bool(infer_cfg.get("return_token_entropy", False))

    logger.info(
        "Sampling parameters - Temperature: %s, Filter: %s, Filter kwargs: %s",
        temperature,
        filter_logits_fn,
        filter_kwargs,
    )

    predictions: list[dict[str, Any]] = []
    total_tokens_generated = 0
    total_tokens_processed = 0
    total_samples_generated = 0
    total_samples_with_eos = 0
    merge_joiner = infer_cfg.get("merge_joiner", " ")

    accelerator.wait_for_everyone()
    generation_start_time = time.time()

    pad_token_id = tokenizer.pad_token_id

    for batch in loader:
        sequences = batch["sequence"]
        if not sequences:
            progress_bar.update(1)
            continue

        prompt_ids_list = []
        for sequence_value in sequences:
            prompt_tokens = prepare_prompt_tokens(
                tokenizer,
                sequence_value,
                sequence_modality,
                condition_cfg,
                max_generation_length,
            )
            prompt_ids_list.append(
                [tokenizer.token_to_id.get(tok, tokenizer.unk_token_id) for tok in prompt_tokens]
            )
        prompt_lengths = [len(ids) for ids in prompt_ids_list]
        max_prompt_len = max(prompt_lengths) if prompt_lengths else 0
        padded_prompts = [
            ids + [pad_token_id] * (max_prompt_len - len(ids))
            for ids in prompt_ids_list
        ]
        prompt_tensor = torch.tensor(
            padded_prompts,
            dtype=torch.long,
            device=accelerator.device,
        )
        prompt_lens = torch.tensor(prompt_lengths, dtype=torch.long, device=accelerator.device)
        context_kwargs = {}
        if tokenizer.sequence_to_structure_use_protein_encoder_context:
            context_kwargs = build_protein_context_batch(
                sequences, max_len=int(configs.model.max_len), device=accelerator.device,
            )
            context_kwargs['is_sequence_to_structure_conditioned'] = torch.ones(
                len(sequences), dtype=torch.bool, device=accelerator.device,
            )

        if tokenizer.max_len and max_prompt_len >= tokenizer.max_len:
            raise ValueError("Prompt length exceeds max sequence length")
        seq_len = max_generation_length
        if tokenizer.max_len:
            max_allowed = max(0, tokenizer.max_len - max_prompt_len)
            seq_len = min(seq_len, max_allowed)

        with torch.inference_mode():
            if context_kwargs:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                context_kwargs = unwrapped.prepare_protein_context_kwargs(prompt_tensor, **context_kwargs)
            generated = model.generate(
                prompts=prompt_tensor,
                seq_len=seq_len,
                eos_token=tokenizer.eos_token_id,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                prompt_lens=prompt_lens,
                **context_kwargs,
            )

        total_tokens_processed += int(generated.shape[0] * generated.shape[1])
        token_metrics = None
        if return_token_probs or return_token_entropy:
            token_metrics = compute_token_metrics(
                model=model,
                token_ids=generated,
                pad_token_id=pad_token_id,
                temperature=temperature,
                return_probs=return_token_probs,
                return_entropy=return_token_entropy,
                model_kwargs=context_kwargs,
            )

        generated_cpu = generated.detach().cpu()

        (
            batch_predictions,
            batch_tokens_generated,
            batch_samples_generated,
            batch_samples_with_eos,
        ) = postprocess_generated_batch(
            generated_cpu,
            prompt_ids_list,
            tokenizer,
            batch["base_row"],
            batch["base_id"],
            batch["generation_index"],
            num_generations_per_sample,
            merge_joiner,
            STRUCTURE_CODEBOOK_SIZE,
            token_metrics=token_metrics,
        )
        predictions.extend(batch_predictions)
        total_tokens_generated += batch_tokens_generated
        total_samples_generated += batch_samples_generated
        total_samples_with_eos += batch_samples_with_eos

        progress_bar.update(1)

    progress_bar.close()

    del model
    _cleanup_memory(accelerator.device)

    accelerator.wait_for_everyone()
    generation_elapsed_time = time.time() - generation_start_time
    logger.info("Generation time: %.2f seconds", generation_elapsed_time)

    token_counts_tensor = torch.tensor(
        [
            total_tokens_generated,
            total_tokens_processed,
            total_samples_generated,
            total_samples_with_eos,
        ],
        device=accelerator.device,
    )
    gathered_counts = accelerator.gather(token_counts_tensor).view(-1, 4)
    total_tokens_generated_all = gathered_counts[:, 0].sum().item()
    total_tokens_processed_all = gathered_counts[:, 1].sum().item()
    total_samples_all = int(gathered_counts[:, 2].sum().item())
    total_eos_all = int(gathered_counts[:, 3].sum().item())

    temp_csv_path = os.path.join(result_dir, f"_temp_rank_{process_index}.csv")
    write_rank_predictions_csv(
        predictions,
        df,
        num_generations_per_sample,
        temp_csv_path,
    )

    accelerator.wait_for_everyone()

    csv_filename = infer_cfg.get("output_csv_filename", "predicted_structures.csv")
    csv_path = os.path.join(result_dir, csv_filename)

    if accelerator.is_main_process:
        all_dfs = []
        for rank in range(num_processes):
            rank_csv_path = os.path.join(result_dir, f"_temp_rank_{rank}.csv")
            if not os.path.exists(rank_csv_path):
                continue
            if os.path.getsize(rank_csv_path) == 0:
                os.remove(rank_csv_path)
                continue
            all_dfs.append(pd.read_csv(rank_csv_path))
            os.remove(rank_csv_path)

        if not all_dfs:
            raise ValueError("No predictions generated across ranks.")

        output_df = pd.concat(all_dfs, ignore_index=True)
        output_df.to_csv(csv_path, index=False)

        logger.info("Saved predictions to %s", csv_path)

        num_generated = total_samples_all
        samples_per_second = num_generated / generation_elapsed_time if generation_elapsed_time > 0 else 0
        real_tps = (
            total_tokens_generated_all / generation_elapsed_time
            if generation_elapsed_time > 0
            else 0
        )
        peak_tps = (
            total_tokens_processed_all / generation_elapsed_time
            if generation_elapsed_time > 0
            else 0
        )
        avg_tokens_per_sample = (
            total_tokens_generated_all / num_generated
            if num_generated > 0
            else 0
        )
        efficiency = (
            total_tokens_generated_all / total_tokens_processed_all * 100
            if total_tokens_processed_all > 0
            else 0
        )
        avg_time_per_sample_ms = (
            generation_elapsed_time / num_generated * 1000
            if num_generated > 0
            else 0
        )

        logger.info("Generation timing statistics:")
        logger.info("  Total samples generated: %d", num_generated)
        logger.info("  Total tokens (real): %s", f"{int(total_tokens_generated_all):,}")
        logger.info("  Total tokens (processed): %s", f"{int(total_tokens_processed_all):,}")
        logger.info("  Token efficiency: %.1f%% (real/processed)", efficiency)
        logger.info("  Average tokens per sample: %.1f", avg_tokens_per_sample)
        logger.info("  Total generation time: %.2f seconds", generation_elapsed_time)
        logger.info("  Throughput: %.2f samples/second", samples_per_second)
        logger.info("  Real TPS: %.2f tokens/second (useful tokens)", real_tps)
        logger.info("  Peak TPS: %.2f tokens/second (model compute)", peak_tps)
        logger.info("  Average time per sample: %.2f ms", avg_time_per_sample_ms)
        logger.info("Sequences containing EOS token: %d/%d", total_eos_all, num_generated)

    accelerator.wait_for_everyone()

    vqvae_decode_cfg = infer_cfg.get("vqvae_decode", {})
    decode_enabled = bool(vqvae_decode_cfg.get("enabled", False))
    if decode_enabled:
        vqvae = _build_vqvae_decoder(
            vqvae_decode_cfg=vqvae_decode_cfg,
            infer_cfg=infer_cfg,
            device=accelerator.device,
            logger=logger,
        )

        decode_batch_size = int(vqvae_decode_cfg.get("batch_size", 16))
        decode_num_workers = int(vqvae_decode_cfg.get("num_workers", 0))
        decode_shuffle = bool(vqvae_decode_cfg.get("shuffle", False))

        compile_cfg = vqvae_decode_cfg.get("compile_model", {})
        if compile_cfg.get("enabled", False):
            logger.warning("vqvae_decode.compile_model is ignored for gcp_vqvae decoding.")

        if accelerator.is_main_process:
            predicted_pdb_dir = os.path.join(result_dir, "predicted_pdbs")
            os.makedirs(predicted_pdb_dir, exist_ok=True)
            paths = [predicted_pdb_dir]
        else:
            paths = [None]

        broadcast_object_list(paths, from_process=0)
        predicted_pdb_dir = paths[0]

        decode_df = pd.read_csv(csv_path)
        if "predicted_structures" not in decode_df.columns:
            raise ValueError("predicted_structures column missing from prediction output")

        records_pred, skipped = build_decode_records(
            decode_df,
            num_generations_per_sample,
            infer_cfg.get("id_column_name"),
            STRUCTURE_CODEBOOK_SIZE,
        )

        if skipped:
            logger.info("Skipping %d samples with empty structure tokens for decode", skipped)
        if not records_pred:
            raise ValueError("No valid structure token rows found for VQ decode")

        decode_dataset = StructureIndicesDataset(records_pred, max_length=vqvae.max_length)
        decode_loader = DataLoader(
            decode_dataset,
            shuffle=decode_shuffle,
            batch_size=decode_batch_size,
            num_workers=decode_num_workers,
        )
        decode_loader = accelerator.prepare(decode_loader)

        progress_bar = tqdm(
            range(0, int(len(decode_loader))),
            leave=True,
            disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
        )
        progress_bar.set_description("Decoding structures")

        for batch in decode_loader:
            with torch.inference_mode():
                indices_batch = batch["indices"].tolist()
                pids = [str(pid) for pid in batch["pid"]]
                results = vqvae.decode(indices_batch, pids=pids, batch_size=len(indices_batch))
                result_pids = results.get("pid", [])
                result_coords = results.get("coords", [])
                result_masks = results.get("mask", [])
                result_plddt = results.get("plddt", [])
                if result_plddt is None:
                    result_plddt = [None] * len(result_pids)
                for pid, coords, mask, plddt in zip(result_pids, result_coords, result_masks, result_plddt):
                    prefix = os.path.join(predicted_pdb_dir, str(pid))
                    save_backbone_pdb_inference(coords, mask, prefix, plddt=plddt)

            progress_bar.update(1)

        progress_bar.close()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence-to-structure inference from amino-acid sequences.")
    parser.add_argument(
        "--config_path",
        "-c",
        help="The location of config file",
        default="./configs/inference_sequence_to_structure_config.yaml",
    )
    args = parser.parse_args()
    main(args)
