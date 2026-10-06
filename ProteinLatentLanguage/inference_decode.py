import argparse
import datetime
import os
import shutil
from typing import Any, Dict, Iterable, List
import csv
import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from box import Box
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.super_model import prepare_models
from utils.log import get_logging
from utils.model import compile_model as compile_without_vq
from utils.utils import load_configs

AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class VQIndicesDataset(Dataset):
    """Dataset that reads VQ index CSV files produced by inference_encode."""

    def __init__(self, csv_path: str, max_length: int) -> None:
        self.data = pd.read_csv(csv_path)
        if 'indices' not in self.data.columns:
            raise ValueError("Expected 'indices' column in the CSV file.")
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        indices_field = str(row['indices']).strip()
        if not indices_field:
            indices = []
        else:
            indices = [int(val) for val in indices_field.split()]

        length = row.get('sequence_length', len(indices))
        if pd.isna(length):
            length = len(indices)
        length = int(length)
        length = max(0, min(length, self.max_length))

        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        padded = indices + [-1] * (self.max_length - len(indices))

        attention_mask = torch.zeros(self.max_length, dtype=torch.bool)
        if length > 0:
            attention_mask[:length] = True

        return {
            'indices': torch.tensor(padded, dtype=torch.long),
            'sequence_length': torch.tensor(length, dtype=torch.long),
            'attention_mask': attention_mask,
        }


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


def _decode_from_indices(vqvae, indices: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    if getattr(vqvae, 'tik_tok_enabled', False):
        raise NotImplementedError("TikTok latent decoding is not yet supported in inference_decode.")
    if getattr(vqvae, 'use_residual_vq', False):
        raise NotImplementedError("Residual VQ decoding is not yet supported in inference_decode.")

    quantizer = getattr(vqvae, 'vector_quantizer', None)
    if quantizer is None or not hasattr(quantizer, 'get_codes_from_indices'):
        raise NotImplementedError("Vector quantizer does not expose get_codes_from_indices; decoding needs custom handling.")

    safe_indices = indices.clone()
    safe_indices[~mask] = 0
    quantized = quantizer.get_codes_from_indices(safe_indices)
    if quantized.dim() == 2:
        quantized = quantized.unsqueeze(0)
    quantized = quantized.to(indices.device)
    quantized = quantized * mask.unsqueeze(-1)

    decoder_mask = mask
    if getattr(vqvae.configs.model.vqvae.decoder, 'enabled', False):
        decoder_embeddings = vqvae.decoder_tail(quantized)
        decoder_attn_mask = None
        if getattr(vqvae, 'decoder_causal', False):
            seq_len = decoder_embeddings.size(1)
            decoder_attn_mask = vqvae.create_causal_mask(seq_len, device=decoder_embeddings.device)
        decoder_output = vqvae.decoder(decoder_embeddings, mask=decoder_mask, attn_mask=decoder_attn_mask)
        if decoder_output.size(1) > vqvae.max_length:
            decoder_output = decoder_output[:, :vqvae.max_length, :]
        decoder_output = vqvae.decoder_head(decoder_output)
    else:
        decoder_output = quantized
        if decoder_output.size(1) > vqvae.max_length:
            decoder_output = decoder_output[:, :vqvae.max_length, :]

    class_logits = decoder_output
    class_probabilities = torch.softmax(class_logits, dim=-1)
    class_indices = class_probabilities.argmax(dim=-1)

    return {
        'quantized_embeddings': quantized,
        'class_probabilities': class_probabilities,
        'class_indices': class_indices,
        'class_logits': class_logits,
    }


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

    override_max_len = infer_cfg.get('max_length', None)
    if override_max_len is not None:
        configs.model.max_len = int(override_max_len)

    dataset = VQIndicesDataset(
        csv_path=infer_cfg.indices_csv_path,
        max_length=configs.model.max_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(infer_cfg.batch_size),
        shuffle=False,
        num_workers=int(infer_cfg.num_workers),
        pin_memory=False,
        drop_last=False,
    )

    logger = get_logging(result_dir, configs)
    logger.info("Starting decode inference over %s entries", len(dataset))

    model = prepare_models(configs, logger, inference=True, decode_only=True)
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
        logger.info("Compiled model for decoder inference (vector quantizer left in eager mode).")

    model, dataloader = accelerator.prepare(model, dataloader)

    num_batches = len(dataloader)

    progress_bar = tqdm(
        range(num_batches),
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
        leave=True,
    )
    progress_bar.set_description("Decode inference")

    records: List[Dict[str, Any]] = []

    vqvae = model.vqvae

    for batch in dataloader:
        with torch.inference_mode():
            indices = batch['indices'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            lengths = batch['sequence_length']

            decoded = _decode_from_indices(vqvae, indices, attention_mask)
            class_probs = decoded['class_probabilities'].detach().cpu()
            class_indices = decoded['class_indices'].detach().cpu()
            class_logits = decoded['class_logits'].detach().cpu()

            for idx_in_batch in range(indices.size(0)):
                length = int(lengths[idx_in_batch].item())
                record: Dict[str, Any] = {
                    'sequence_length': length,
                }

                probs_np = class_probs[idx_in_batch, :length].numpy().astype(np.float32)
                indices_np = class_indices[idx_in_batch, :length].numpy().astype(np.int32)
                decoded_sequence = "".join(
                    AMINO_ACID_ALPHABET[idx] if 0 <= idx < len(AMINO_ACID_ALPHABET) else 'X'
                    for idx in indices_np
                )
                record['class_probabilities'] = probs_np
                record['decoded_sequence'] = decoded_sequence

                records.append(record)

            progress_bar.update(1)

    # end progress_bar
    progress_bar.close()

    accelerator.wait_for_everyone()
    gathered = accelerator.gather_for_metrics(records, use_gather_object=True)

    if accelerator.is_main_process:
        combined = _flatten_records(gathered)

        csv_path = os.path.join(result_dir, 'decoded_sequences.csv')

        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['sequence_length', 'decoded_sequence'])
            for rec in combined:
                csv_writer.writerow([
                    rec['sequence_length'],
                    rec['decoded_sequence'],
                ])

        logger.info("Saved decoded sequences to %s", csv_path)

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode VQ indices using the Protein VQ-VAE.")
    parser.add_argument(
        "--config_path",
        "-c",
        default="configs/inference_decode_config.yaml",
        help="Path to the decode inference configuration file.",
    )
    main(parser.parse_args())
