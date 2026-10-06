import csv
import os
import re
from typing import Any, Dict

import pandas as pd
import torch
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inference.sequence_to_structure import (
    StructureIndicesDataset,
    parse_structure_indices,
    save_backbone_pdb_inference,
)


STRUCTURE_CODEBOOK_SIZE = 4096


def _build_vqvae_decoder(
    vqvae_decode_cfg: Dict[str, Any],
    infer_cfg,
    device: torch.device,
    logger,
):
    from gcp_vqvae import GCPVQVAE

    trained_model_dir = vqvae_decode_cfg.get("trained_model_dir")
    if not trained_model_dir:
        raise ValueError("vqvae_decode.trained_model_dir is required for structure decoding")

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


def _parse_structure_indices_for_decode(
    value: Any,
    merge_joiner: str,
    codebook_size: int,
) -> list[int]:
    indices = parse_structure_indices(value, codebook_size)
    if indices:
        return indices

    raw = str(value).strip()
    if not raw:
        return []

    if merge_joiner and merge_joiner not in (" ", "\t", "\n"):
        indices = parse_structure_indices(raw.replace(merge_joiner, " "), codebook_size)
        if indices:
            return indices

    matches = re.findall(r"-?\d+(?=_3D)", raw)
    parsed: list[int] = []
    for match in matches:
        idx = int(match)
        if -1 <= idx < codebook_size:
            parsed.append(idx)
    return parsed


def _build_denovo_decode_records(
    records: list[dict[str, Any]],
    seq_column: str,
    merge_joiner: str,
    codebook_size: int,
) -> tuple[list[dict[str, Any]], int]:
    decode_records: list[dict[str, Any]] = []
    skipped = 0
    for rec in records:
        indices = _parse_structure_indices_for_decode(
            rec.get(seq_column, ""),
            merge_joiner=merge_joiner,
            codebook_size=codebook_size,
        )
        if not indices:
            skipped += 1
            continue
        decode_records.append(
            {
                "pid": str(rec.get("sample_index")),
                "indices": indices,
            }
        )
    return decode_records, skipped


def _decode_denovo_structures(
    *,
    data_type: str,
    infer_cfg,
    accelerator,
    logger,
    result_dir: str,
    csv_path: str,
    seq_column: str,
    merge_joiner: str,
) -> None:
    if data_type != "structure":
        return

    vqvae_decode_cfg = infer_cfg.get("vqvae_decode", {})
    decode_enabled = bool(vqvae_decode_cfg.get("enabled", True))
    if not decode_enabled:
        logger.info("Structure modality detected, but vqvae_decode.enabled is false; skipping PDB decoding.")
        return

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
    if seq_column not in decode_df.columns:
        raise ValueError(f"{seq_column} column missing from de novo output for VQ decode")

    records_pred, skipped = _build_denovo_decode_records(
        decode_df.to_dict(orient="records"),
        seq_column=seq_column,
        merge_joiner=merge_joiner,
        codebook_size=STRUCTURE_CODEBOOK_SIZE,
    )
    if skipped:
        logger.info("Skipping %d samples with empty/invalid structure tokens for decode.", skipped)
    if not records_pred:
        raise ValueError("No valid structure sequences found for VQ-VAE decoding.")

    decode_dataset = StructureIndicesDataset(records_pred, max_length=vqvae.max_length)
    decode_loader = DataLoader(
        decode_dataset,
        shuffle=decode_shuffle,
        batch_size=decode_batch_size,
        num_workers=decode_num_workers,
    )
    decode_loader = accelerator.prepare(decode_loader)

    decode_bar = tqdm(
        range(0, int(len(decode_loader))),
        leave=True,
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
    )
    decode_bar.set_description("Decoding structures")

    aa_rows: list[dict[str, str]] = []

    for batch in decode_loader:
        with torch.inference_mode():
            indices_batch = batch["indices"].tolist()
            pids = [str(pid) for pid in batch["pid"]]
            results = vqvae.decode(indices_batch, pids=pids, batch_size=len(indices_batch))
            result_pids = results.get("pid", [])
            result_coords = results.get("coords", [])
            result_masks = results.get("mask", [])
            result_plddt = results.get("plddt", [])
            result_aa = results.get("AA", [])
            if result_plddt is None:
                result_plddt = [None] * len(result_pids)
            if result_aa is None:
                result_aa = [None] * len(result_pids)
            for pid, coords, mask, plddt, aa in zip(result_pids, result_coords, result_masks, result_plddt, result_aa):
                prefix = os.path.join(predicted_pdb_dir, str(pid))
                aa_sequence = str(aa) if aa is not None else None
                save_backbone_pdb_inference(coords, mask, prefix, plddt=plddt, aa_sequence=aa_sequence)
                if aa_sequence:
                    aa_rows.append(
                        {
                            "pid": str(pid),
                            "pdb_file": f"{pid}.pdb",
                            "AA": aa_sequence,
                        }
                    )
        decode_bar.update(1)

    decode_bar.close()

    seq_temp_path = os.path.join(
        result_dir,
        f"_temp_recovered_aa_rank_{accelerator.process_index}.csv",
    )
    with open(seq_temp_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for row in aa_rows:
            writer.writerow([row["pid"], row["pdb_file"], row["AA"]])

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_rows: list[dict[str, str]] = []
        for rank in range(accelerator.num_processes):
            rank_path = os.path.join(result_dir, f"_temp_recovered_aa_rank_{rank}.csv")
            if not os.path.exists(rank_path):
                continue
            with open(rank_path, newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) != 3:
                        continue
                    merged_rows.append({"pid": row[0], "pdb_file": row[1], "AA": row[2]})
            os.remove(rank_path)

        if merged_rows:
            merged_rows.sort(
                key=lambda x: (
                    0,
                    int(x["pid"]),
                )
                if x["pid"].isdigit()
                else (1, x["pid"])
            )
            seq_csv_path = os.path.join(result_dir, "predicted_sequences.csv")
            with open(seq_csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pid", "pdb_file", "AA"])
                for row in merged_rows:
                    writer.writerow([row["pid"], row["pdb_file"], row["AA"]])
            logger.info("Saved recovered amino-acid sequences to %s", seq_csv_path)
        else:
            logger.info("No recovered amino-acid sequences returned by decoder; skipped sequence CSV.")

    accelerator.wait_for_everyone()
