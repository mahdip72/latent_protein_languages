import argparse
import csv
import datetime
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ProteinDataset
from models.super_model import prepare_models
from utils.log import get_logging
from utils.model import compile_model as compile_without_vq
from utils.utils import load_configs


def _load_yaml_box(config_path: str) -> Box:
    with open(config_path) as handle:
        data = yaml.full_load(handle)
    return Box(data)


def _merge_vq_config(base_cfg: Dict[str, Any], trained_dir: str) -> None:
    vq_cfg = (
        base_cfg.get('model', {})
        .get('vqvae', {})
        .get('vector_quantization', {})
    )
    vq_type = vq_cfg.get('type', None)
    if not vq_type:
        return

    candidate_paths = [
        os.path.join(trained_dir, f"{vq_type}.yaml"),
        os.path.join("configs", f"{vq_type}.yaml"),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            with open(candidate) as handle:
                extra = yaml.full_load(handle) or {}
            extra['enabled'] = vq_cfg.get('enabled', extra.get('enabled', True))
            extra['type'] = vq_type
            vq_cfg.update(extra)
            break


def _load_trained_run_configs(trained_dir: str, config_filename: str) -> Box:
    config_path = os.path.join(trained_dir, config_filename)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config not found at {config_path}")

    with open(config_path) as handle:
        base_cfg = yaml.full_load(handle)
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Malformed config file: {config_path}")

    _merge_vq_config(base_cfg, trained_dir)
    return load_configs(base_cfg, inference=True)


def _prepare_tokenizer(configs: Box):
    model_type = configs.model.protein_encoder.model_type
    model_name = configs.model.protein_encoder.model_name

    if model_type == 'esm_v2':
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    if model_type == 'esmc':
        from esm.models.esmc import ESMC
        return ESMC.from_pretrained(model_name).tokenizer
    raise ValueError(f"Unsupported protein encoder type: {model_type}")


def _truncate_dataset(dataset: ProteinDataset, max_samples: Optional[int]) -> None:
    if max_samples is None:
        return
    if max_samples <= 0:
        raise ValueError("max_task_samples must be a positive integer when provided.")
    if max_samples < len(dataset.protein_sequences):
        dataset.protein_sequences = dataset.protein_sequences[:max_samples]


def _create_result_dir(
    accelerator: Accelerator,
    infer_cfg: Box,
    config_path: str,
) -> str:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy(config_path, result_dir)
        paths = [result_dir]
    else:
        paths = [None]

    broadcast_object_list(paths, from_process=0)
    return paths[0]


def _flatten_records(records: Iterable[Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for entry in records:
        if isinstance(entry, list):
            flattened.extend(entry)
        else:
            flattened.append(entry)
    return flattened


def main(args: argparse.Namespace) -> None:
    infer_cfg = _load_yaml_box(args.config_path)

    dataloader_cfg = DataLoaderConfiguration(
        non_blocking=True,
        even_batches=False,
    )
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_cfg,
    )

    result_dir = _create_result_dir(accelerator, infer_cfg, args.config_path)

    trained_dir = infer_cfg.trained_model_dir
    configs = _load_trained_run_configs(trained_dir, infer_cfg.config_filename)

    max_task_samples = infer_cfg.get('max_task_samples', None)
    if max_task_samples is not None:
        max_task_samples = int(max_task_samples)

    override_max_len = infer_cfg.get('max_length', None)
    if override_max_len is not None:
        configs.model.max_len = int(override_max_len)

    tokenizer = _prepare_tokenizer(configs)
    dataset = ProteinDataset(
        csv_file=infer_cfg.data_path,
        tokenizer=tokenizer,
        configs=configs,
        protein_column_name=infer_cfg.get('protein_column_name', 'Amino Acid Sequence'),
        mode='inference',
    )
    _truncate_dataset(dataset, max_task_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=int(infer_cfg.batch_size),
        shuffle=bool(infer_cfg.shuffle),
        num_workers=int(infer_cfg.num_workers),
        pin_memory=False,
        drop_last=False,
    )

    logger = get_logging(result_dir, configs)
    logger.info("Starting encode inference over %s samples", len(dataset))

    model = prepare_models(configs, logger, inference=True)
    model.eval()

    checkpoint_path = os.path.join(trained_dir, infer_cfg.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    load_log = model.load_state_dict(state_dict, strict=False)
    logger.info("Loading checkpoint log: %s", load_log)

    compile_cfg = infer_cfg.get('compile_model', {})
    if compile_cfg.get('enabled', False):
        model = compile_without_vq(model, mode=compile_cfg.get('mode', None))
        logger.info("Compiled model for inference (vector quantizer left in eager mode).")

    model, dataloader = accelerator.prepare(model, dataloader)

    num_batches = len(dataloader)

    progress_bar = tqdm(
        range(num_batches),
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
        leave=True,
    )
    progress_bar.set_description("Encode inference")

    records: List[Dict[str, Any]] = []

    for batch in dataloader:
        with torch.inference_mode():
            batch['input_ids'] = batch['input_ids'].to(accelerator.device)
            batch['attention_mask'] = batch['attention_mask'].to(accelerator.device)

            outputs = model(batch)
            indices = outputs['indices']
            attention_mask = batch['attention_mask']
            sequences = batch['original_sequence']

            indices_cpu = indices.detach().cpu()
            mask_cpu = attention_mask.detach().cpu()

            for idx_in_batch, sequence in enumerate(sequences):
                valid_length = int(mask_cpu[idx_in_batch].sum().item())
                trimmed_indices = indices_cpu[idx_in_batch, :valid_length].tolist()

                records.append({
                    'sequence_length': valid_length,
                    'indices': trimmed_indices,
                    'sequence': sequence,
                })

            progress_bar.update(1)

    # end progress_bar
    progress_bar.close()

    accelerator.wait_for_everyone()
    gathered = accelerator.gather_for_metrics(records, use_gather_object=True)
    if accelerator.is_main_process:
        combined = _flatten_records(gathered)
        csv_path = os.path.join(result_dir, infer_cfg.get('vq_indices_csv_filename', 'vq_indices.csv'))
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['sequence_length', 'indices', 'sequence'])
            for rec in combined:
                indices_str = ' '.join(str(v) for v in rec['indices'])
                writer.writerow([
                    rec['sequence_length'],
                    indices_str,
                    rec['sequence'],
                ])
        logger.info("Saved VQ indices to %s", csv_path)

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode sequences through the Protein VQ-VAE.")
    parser.add_argument(
        "--config_path",
        "-c",
        default="configs/inference_encode_config.yaml",
        help="Path to the encode inference configuration file.",
    )
    main(parser.parse_args())
