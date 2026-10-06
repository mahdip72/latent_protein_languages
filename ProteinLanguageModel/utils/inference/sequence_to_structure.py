import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from data.dataset import ProteinTokenizer
from utils.model import (
    _find_tokenizer_vocab_path,
    _load_tokenizer_vocab,
    _remap_embedding_weights,
)


_E1_VOCAB_CACHE: dict[str, int] | None = None


def resolve_condition_enabled(condition_cfg: Dict[str, Any], name: str) -> bool:
    return bool(condition_cfg.get(name, {}).get("enabled", False))


def resolve_length_override(condition_cfg: Dict[str, Any]) -> int | None:
    value = condition_cfg.get("length", {}).get("fixed_value", None)
    if value is None:
        return None
    return int(value)


def select_active_conditions(
    tokenizer: ProteinTokenizer,
    condition_cfg: Dict[str, Any],
) -> list:
    """Select active conditions in tokenizer order for inference."""
    active_conditions = []
    for cond in tokenizer.conditions:
        if cond.name == "sequence_to_structure":
            active_conditions.append(cond)
        elif cond.name == "c2n" and resolve_condition_enabled(condition_cfg, "c2n"):
            active_conditions.append(cond)
        elif cond.name == "length" and resolve_condition_enabled(condition_cfg, "length"):
            active_conditions.append(cond)
    return active_conditions


def validate_condition_support(
    tokenizer: ProteinTokenizer,
    condition_cfg: Dict[str, Any],
) -> None:
    if not tokenizer.sequence_to_structure_enabled:
        raise ValueError("sequence_to_structure is required in the training config for this inference script")
    if resolve_condition_enabled(condition_cfg, "c2n"):
        if not any(c.name == "c2n" for c in tokenizer.conditions):
            raise ValueError("Inference config enables C2N but tokenizer was built without C2N")
    if resolve_condition_enabled(condition_cfg, "length"):
        if not any(c.name == "length" for c in tokenizer.conditions):
            raise ValueError("Inference config enables length but tokenizer was built without length")


def _get_e1_vocab() -> dict[str, int]:
    global _E1_VOCAB_CACHE
    if _E1_VOCAB_CACHE is None:
        try:
            from E1.tokenizer import get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "protein_encoder_context with model_type='e1' requires the E1 package."
            ) from exc
        _E1_VOCAB_CACHE = get_tokenizer().get_vocab()
    return _E1_VOCAB_CACHE


def build_protein_context_inputs(
    protein_sequence: str,
    max_len: int,
) -> dict[str, torch.Tensor]:
    seq = normalize_raw_value(protein_sequence).upper()
    if "," in seq:
        raise ValueError(
            "E1 context integration expects single sequences only; found a comma-separated entry."
        )

    seq = seq[:max_len]
    seq_len = len(seq)
    vocab = _get_e1_vocab()

    pad = vocab["<pad>"]
    bos = vocab["<bos>"]
    eos = vocab["<eos>"]
    one = vocab["1"]
    two = vocab["2"]
    x_id = vocab.get("X", pad)

    total_len = max_len + 4
    input_ids = torch.full((total_len,), pad, dtype=torch.long)
    within_seq_position_ids = torch.full((total_len,), -1, dtype=torch.long)
    global_position_ids = torch.full((total_len,), -1, dtype=torch.long)
    sequence_ids = torch.full((total_len,), -1, dtype=torch.long)

    residue_ids = [vocab.get(ch, x_id) for ch in seq]
    tokens = [bos, one, *residue_ids, two, eos]
    valid_len = len(tokens)

    input_ids[:valid_len] = torch.tensor(tokens, dtype=torch.long)
    positions = torch.arange(valid_len, dtype=torch.long)
    within_seq_position_ids[:valid_len] = positions
    global_position_ids[:valid_len] = positions
    sequence_ids[:valid_len] = 0

    residue_attention_mask = torch.zeros(max_len, dtype=torch.bool)
    if seq_len > 0:
        residue_attention_mask[:seq_len] = True

    return {
        "protein_encoder_context_input_ids": input_ids,
        "protein_encoder_context_within_seq_position_ids": within_seq_position_ids,
        "protein_encoder_context_global_position_ids": global_position_ids,
        "protein_encoder_context_sequence_ids": sequence_ids,
        "protein_encoder_context_attention_mask": residue_attention_mask,
    }


def build_protein_context_batch(
    sequences: Iterable[str],
    max_len: int,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    rows = [build_protein_context_inputs(sequence, max_len) for sequence in sequences]
    if not rows:
        return {}

    batch = {
        key: torch.stack([row[key] for row in rows], dim=0)
        for key in rows[0]
    }
    if device is not None:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
    return batch


def prepare_prompt_tokens(
    tokenizer: ProteinTokenizer,
    sequence_value: Any,
    sequence_modality: str,
    condition_cfg: Dict[str, Any],
    max_generation_length: int,
) -> list[str]:
    """Build prompt tokens for sequence-to-structure generation."""
    sequence_tokens = tokenizer._parse_tokens(sequence_value, sequence_modality)

    active_conditions = select_active_conditions(tokenizer, condition_cfg)
    for cond in active_conditions:
        sequence_tokens = cond.apply_transform(sequence_tokens)

    max_len = tokenizer.max_len or 0
    length_override = resolve_length_override(condition_cfg)
    prefix_tokens: list[str] = []
    if tokenizer.sequence_to_structure_use_protein_encoder_context:
        if len(sequence_tokens) > max_len:
            sequence_tokens = sequence_tokens[:max_len]

        if length_override is not None:
            max_allowed = max(1, max_len)
            if length_override > max_allowed:
                raise ValueError(
                    f"length.fixed_value={length_override} exceeds max allowed {max_allowed} for tokenizer"
                )

        for cond in active_conditions:
            content_len = len(sequence_tokens)
            if cond.name == "length" and length_override is not None:
                content_len = length_override
            tok = cond.get_token(content_len)
            if tok:
                prefix_tokens.append(tok)

        return ["<BOS>"] + prefix_tokens + ["<BO3D>"]

    prefix_count = len([cond for cond in active_conditions if cond.token is not None])
    max_seq_len = max_len - max_generation_length - (4 + prefix_count)
    if max_seq_len < 0:
        max_seq_len = 0
    if len(sequence_tokens) > max_seq_len:
        sequence_tokens = sequence_tokens[:max_seq_len]

    sequence_tokens = [t if t in tokenizer.token_to_id else "<UNK>" for t in sequence_tokens]

    if length_override is not None:
        max_allowed = max(1, max_len - 4)
        if length_override > max_allowed:
            raise ValueError(
                f"length.fixed_value={length_override} exceeds max allowed {max_allowed} for tokenizer"
            )

    for cond in active_conditions:
        content_len = len(sequence_tokens)
        if cond.name == "length" and length_override is not None:
            content_len = length_override
        tok = cond.get_token(content_len)
        if tok:
            prefix_tokens.append(tok)

    return ["<BOS>"] + prefix_tokens + ["<BOP>"] + sequence_tokens + ["<EOP>"]


def extract_structure_tokens(tokens: list[str]) -> list[str]:
    """Extract structure tokens between BO3D and EO3D/EOS markers."""
    try:
        start = tokens.index("<BO3D>")
    except ValueError:
        return [t for t in tokens if t.endswith("_3D")]

    end = len(tokens)
    for idx in range(start + 1, len(tokens)):
        if tokens[idx] in ("<EO3D>", "<EOS>"):
            end = idx
            break

    return [t for t in tokens[start + 1:end] if t.endswith("_3D")]


def normalize_raw_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_structure_indices(value: Any, codebook_size: int) -> list[int]:
    raw = normalize_raw_value(value)
    if not raw:
        return []
    indices: list[int] = []
    for token in raw.split():
        token = token.strip()
        if not token:
            continue
        if token.endswith("_3D"):
            token = token[:-3]
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx < -1 or idx >= codebook_size:
            continue
        indices.append(idx)
    return indices


def structure_tokens_to_indices(tokens: list[str], codebook_size: int) -> list[str]:
    indices: list[str] = []
    for token in tokens:
        if token.endswith("_3D"):
            token = token[:-3]
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx < -1 or idx >= codebook_size:
            continue
        indices.append(str(idx))
    return indices


def prepare_inference_state_dict(
    state_dict: dict,
    model: torch.nn.Module,
    tokenizer: ProteinTokenizer,
    checkpoint_path: str,
    logger,
) -> dict:
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("transformer."):
            new_key = key.replace("transformer.", "transformer.net.", 1)
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value

    current_state = model.state_dict()
    vocab_path = _find_tokenizer_vocab_path(checkpoint_path, None)
    if vocab_path:
        old_token_to_id = _load_tokenizer_vocab(vocab_path)
        new_token_to_id = getattr(tokenizer, "token_to_id", None)
        if old_token_to_id and new_token_to_id:
            logger.info("Using tokenizer vocab from %s for embedding remap.", vocab_path)
            mapped_state_dict = _remap_embedding_weights(
                mapped_state_dict,
                current_state,
                old_token_to_id,
                new_token_to_id,
                logger,
            )
        else:
            logger.info("Tokenizer vocab file missing token mappings at %s.", vocab_path)
    else:
        logger.info("No tokenizer_vocab.yaml found for embedding remap; loading weights as-is.")

    filtered_state = {}
    skipped_shape = []
    for key, tensor in mapped_state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != tensor.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = tensor

    if skipped_shape:
        logger.info(
            "Skipped %d keys due to shape mismatch (sample: %s).",
            len(skipped_shape),
            skipped_shape[:5],
        )

    return filtered_state


def _extract_structure_tokens_and_positions(full_tokens: list[str]) -> tuple[list[str], list[int]]:
    """Return structure tokens and their positions within the full token sequence."""
    structure_tokens: list[str] = []
    positions: list[int] = []
    try:
        start = full_tokens.index("<BO3D>")
        end = len(full_tokens)
        for idx in range(start + 1, len(full_tokens)):
            if full_tokens[idx] in ("<EO3D>", "<EOS>"):
                end = idx
                break
        for idx in range(start + 1, end):
            tok = full_tokens[idx]
            if tok.endswith("_3D"):
                structure_tokens.append(tok)
                positions.append(idx)
    except ValueError:
        for idx, tok in enumerate(full_tokens):
            if tok.endswith("_3D"):
                structure_tokens.append(tok)
                positions.append(idx)
    return structure_tokens, positions


def _align_token_values(
    values: list[float] | None,
    prompt_len: int,
    full_len: int,
    includes_prompt: bool,
) -> list[float] | None:
    """Align token-level values to the full token list, padding prompt with NaNs if needed."""
    if values is None:
        return None
    if len(values) == full_len:
        return values
    if not includes_prompt and len(values) + prompt_len == full_len:
        return [float("nan")] * prompt_len + values
    return None


def _format_metric_values(values: list[float]) -> str:
    """Format a list of floats as a space-separated string with fixed precision."""
    return " ".join(f"{val:.6f}" if val == val else "nan" for val in values)


def _compute_metric_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
    return_probs: bool,
    return_entropy: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute per-token probabilities and entropy from logits."""
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_mask = target_ids != pad_token_id
    token_log_probs = token_log_probs.masked_fill(~target_mask, float("nan"))

    token_probs = None
    token_entropy = None
    if return_probs:
        token_probs = token_log_probs.exp()
    if return_entropy:
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = entropy.masked_fill(~target_mask, float("nan"))
        token_entropy = entropy

    pad = torch.full((token_log_probs.shape[0], 1), float("nan"), device=token_log_probs.device)
    if token_probs is not None:
        token_probs = torch.cat([pad, token_probs], dim=1)
    if token_entropy is not None:
        token_entropy = torch.cat([pad, token_entropy], dim=1)

    return token_probs, token_entropy


def compute_token_metrics(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_probs: bool,
    return_entropy: bool,
    model_kwargs: dict[str, torch.Tensor] | None = None,
) -> dict[str, list[list[float]] | None]:
    """Compute raw and temperature-shifted token metrics for generated sequences."""
    if not (return_probs or return_entropy):
        return {
            "probs_raw": None,
            "probs_temp": None,
            "entropy_raw": None,
            "entropy_temp": None,
        }

    temp = float(temperature) if temperature and temperature > 0 else 1.0

    base_model = model
    visited = set()
    while hasattr(base_model, "module") and id(base_model) not in visited:
        visited.add(id(base_model))
        base_model = base_model.module

    forward_kwargs = {}
    if bool(getattr(base_model, "sequence_to_structure_use_protein_encoder_context", False)):
        if model_kwargs is None:
            raise ValueError(
                "Token metrics for protein_encoder_context checkpoints require model_kwargs."
            )
        prepare_context = getattr(base_model, "prepare_protein_context_kwargs", None)
        if not callable(prepare_context):
            raise ValueError("Model is missing prepare_protein_context_kwargs().")
        forward_kwargs = prepare_context(token_ids, **model_kwargs)

    transformer = getattr(base_model, "transformer", None)
    if transformer is None:
        raise ValueError("Model is missing transformer for token scoring.")
    net = getattr(transformer, "net", transformer)

    mask = token_ids != pad_token_id
    with torch.inference_mode():
        logits = net(token_ids, mask=mask, **forward_kwargs)
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("output", logits))
        if not isinstance(logits, torch.Tensor):
            raise ValueError("Unexpected logits output when computing token metrics.")

        prepend_embeds = forward_kwargs.get("prepend_embeds")
        if isinstance(prepend_embeds, torch.Tensor):
            prepend_len = int(prepend_embeds.shape[1])
            if prepend_len > 0:
                if logits.shape[1] < prepend_len:
                    raise ValueError(
                        f"Logits length {logits.shape[1]} is smaller than prepend length {prepend_len}."
                    )
                logits = logits[:, prepend_len:, :]

        logits = logits.float()
        target_ids = token_ids[:, 1:]

        raw_logits = logits[:, :-1, :]
        temp_logits = raw_logits / temp

        raw_probs, raw_entropy = _compute_metric_from_logits(
            raw_logits,
            target_ids,
            pad_token_id,
            return_probs,
            return_entropy,
        )
        temp_probs, temp_entropy = _compute_metric_from_logits(
            temp_logits,
            target_ids,
            pad_token_id,
            return_probs,
            return_entropy,
        )

    return {
        "probs_raw": raw_probs.detach().cpu().tolist() if raw_probs is not None else None,
        "probs_temp": temp_probs.detach().cpu().tolist() if temp_probs is not None else None,
        "entropy_raw": raw_entropy.detach().cpu().tolist() if raw_entropy is not None else None,
        "entropy_temp": temp_entropy.detach().cpu().tolist() if temp_entropy is not None else None,
    }


def postprocess_generated_batch(
    generated_cpu: torch.Tensor,
    prompt_ids_list: List[List[int]],
    tokenizer: ProteinTokenizer,
    batch_base_rows: List[dict],
    batch_base_ids: List[Any],
    batch_generation_indices: List[int],
    num_generations_per_sample: int,
    merge_joiner: str,
    codebook_size: int,
    token_metrics: dict[str, list[list[float]] | None] | None = None,
) -> Tuple[List[dict], int, int, int]:
    predictions: List[dict] = []
    tokens_generated = 0
    samples_generated = 0
    samples_with_eos = 0

    for row_idx in range(len(batch_base_rows)):
        token_ids = generated_cpu[row_idx].tolist()
        tokens = [tokenizer.id_to_token.get(int(tid), "<UNK>") for tid in token_ids]
        prompt_ids = prompt_ids_list[row_idx]
        includes_prompt = (
            len(token_ids) >= len(prompt_ids)
            and token_ids[: len(prompt_ids)] == prompt_ids
        )
        if includes_prompt:
            full_tokens = tokens
        else:
            prompt_tokens = [
                tokenizer.id_to_token.get(int(tid), "<UNK>")
                for tid in prompt_ids
            ]
            full_tokens = prompt_tokens + tokens

        structure_tokens, structure_positions = _extract_structure_tokens_and_positions(full_tokens)
        structure_indices = structure_tokens_to_indices(
            structure_tokens,
            codebook_size,
        )
        tokens_generated += len(structure_indices)
        samples_generated += 1
        if tokenizer.eos_token_id in token_ids:
            samples_with_eos += 1

        output_row = dict(batch_base_rows[row_idx])
        if num_generations_per_sample > 1:
            base_id = batch_base_ids[row_idx]
            generation_index = batch_generation_indices[row_idx]
            prediction_id = f"{base_id}__gen_{generation_index}"
            output_row["prediction_id"] = prediction_id
            output_row["generation_index"] = generation_index
        output_row["predicted_structures"] = merge_joiner.join(structure_indices)
        if token_metrics is not None:
            prompt_len = len(prompt_ids)
            metric_map = {
                "predicted_structure_token_probs_raw": token_metrics.get("probs_raw"),
                "predicted_structure_token_probs_temp": token_metrics.get("probs_temp"),
                "predicted_structure_token_entropy_raw": token_metrics.get("entropy_raw"),
                "predicted_structure_token_entropy_temp": token_metrics.get("entropy_temp"),
            }
            for col_name, values in metric_map.items():
                if values is None:
                    continue
                aligned = _align_token_values(values[row_idx], prompt_len, len(full_tokens), includes_prompt)
                if aligned is None:
                    continue
                struct_vals = [aligned[pos] for pos in structure_positions]
                output_row[col_name] = _format_metric_values(struct_vals)
        predictions.append(output_row)

    return predictions, tokens_generated, samples_generated, samples_with_eos


def write_rank_predictions_csv(
    predictions: List[dict],
    base_df: pd.DataFrame,
    num_generations_per_sample: int,
    csv_path: str,
) -> None:
    if predictions:
        temp_df = pd.DataFrame(predictions)
        output_columns = base_df.columns.tolist()
        extra_columns = []
        if num_generations_per_sample > 1:
            extra_columns.extend(["prediction_id", "generation_index"])
        extra_columns.append("predicted_structures")
        for col in extra_columns:
            if col not in output_columns:
                output_columns.append(col)
        for col in temp_df.columns:
            if col not in output_columns:
                output_columns.append(col)
        temp_df = temp_df.reindex(columns=output_columns)
        temp_df.to_csv(csv_path, index=False)
    else:
        open(csv_path, "w").close()


def build_decode_records(
    decode_df: pd.DataFrame,
    num_generations_per_sample: int,
    id_column: Optional[str],
    codebook_size: int,
) -> Tuple[List[dict], int]:
    records_pred: List[dict] = []
    skipped = 0
    for row_idx, row in decode_df.iterrows():
        pid = None
        if num_generations_per_sample > 1 and "prediction_id" in decode_df.columns:
            pred_id_val = row.get("prediction_id")
            if pred_id_val is not None and not (isinstance(pred_id_val, float) and math.isnan(pred_id_val)):
                pid = pred_id_val
        if pid is None:
            pid = row.get(id_column) if id_column and id_column in decode_df.columns else row_idx
        if pid is None or (isinstance(pid, float) and math.isnan(pid)):
            pid = row_idx
        pid = str(pid)

        pred_indices = parse_structure_indices(row["predicted_structures"], codebook_size)
        if not pred_indices:
            skipped += 1
            continue
        records_pred.append({"pid": pid, "indices": pred_indices})

    return records_pred, skipped


class StructureIndicesDataset(Dataset):
    """Dataset wrapper for decoding structure indices into coordinates."""

    def __init__(self, records: list[dict[str, Any]], max_length: int):
        self.records = records
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        indices = list(rec["indices"])
        if len(indices) > self.max_length:
            indices = indices[: self.max_length]
        pad_length = self.max_length - len(indices)
        padded_indices = indices + [-1] * pad_length

        return {
            "pid": rec["pid"],
            "indices": torch.tensor(padded_indices, dtype=torch.long),
        }


def save_backbone_pdb_inference(
    coords: torch.Tensor,
    masks: torch.Tensor,
    save_path_prefix: str,
    atom_names: tuple[str, ...] = ("N", "CA", "C"),
    chain_id: str = "A",
    plddt: torch.Tensor | None = None,
    aa_sequence: str | None = None,
) -> None:
    """Save a single backbone PDB file without sample suffix."""
    aa_to_resname = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "U": "SEC",
        "O": "PYL",
        "B": "ASX",
        "Z": "GLX",
        "J": "XLE",
        "X": "UNK",
    }

    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        masks = masks.unsqueeze(0)

    batch_size, length = coords.shape[:2]
    for b in range(batch_size):
        if save_path_prefix.lower().endswith(".pdb"):
            out_path = save_path_prefix
        else:
            out_path = f"{save_path_prefix}.pdb"
        with open(out_path, "w") as handle:
            serial = 1
            aa_idx = 0
            for r in range(length):
                if not masks[b, r].item():
                    continue
                resname = "ALA"
                if aa_sequence:
                    if aa_idx < len(aa_sequence):
                        resname = aa_to_resname.get(aa_sequence[aa_idx].upper(), "UNK")
                    aa_idx += 1
                b_factor = 0.00
                if plddt is not None:
                    value = plddt[b, r].item() if plddt.dim() == 2 else plddt[r].item()
                    if value == value:
                        if value <= 1.5:
                            value = value * 100.0
                        b_factor = max(0.0, min(100.0, float(value)))
                for atom_idx, atom_name in enumerate(atom_names):
                    x, y, z = coords[b, r, atom_idx].tolist()
                    line = (
                        f"ATOM  {serial:5d} {atom_name:>4} {resname:>3} {chain_id:>1}"
                        f"{r + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}"
                        f"           {atom_name[0]:>2}\n"
                    )
                    handle.write(line)
                    serial += 1
            handle.write("END\n")


class SequenceToStructureDataset(Dataset):
    """Dataset for sequence-to-structure inference with duplicated generations."""

    def __init__(
        self,
        df,
        sequence_column: str,
        num_generations_per_sample: int,
        id_column: Optional[str] = None,
        sequence_values: Optional[Iterable[str]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sequence_column = sequence_column
        self.id_column = id_column
        self.num_generations_per_sample = max(1, int(num_generations_per_sample))

        if sequence_values is None:
            self.sequence_values = [
                normalize_raw_value(self.df.at[idx, sequence_column])
                for idx in range(len(self.df))
            ]
        else:
            values = list(sequence_values)
            if len(values) != len(self.df):
                raise ValueError("sequence_values length must match df length.")
            self.sequence_values = values

    def __len__(self) -> int:
        return len(self.df) * self.num_generations_per_sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx = idx // self.num_generations_per_sample
        gen_idx = idx % self.num_generations_per_sample

        row = self.df.iloc[row_idx]
        base_row = row.to_dict()
        if self.id_column and self.id_column in self.df.columns:
            base_id = base_row.get(self.id_column)
        else:
            base_id = row_idx
        if base_id is None or (isinstance(base_id, float) and math.isnan(base_id)):
            base_id = row_idx

        return {
            "sequence": self.sequence_values[row_idx],
            "base_row": base_row,
            "base_id": base_id,
            "generation_index": gen_idx,
        }


def sequence_to_structure_collate(batch: list[dict[str, Any]]) -> dict[str, List[Any]]:
    return {
        "sequence": [item["sequence"] for item in batch],
        "base_row": [item["base_row"] for item in batch],
        "base_id": [item["base_id"] for item in batch],
        "generation_index": [item["generation_index"] for item in batch],
    }


class SequenceRowDataset(Dataset):
    """Dataset for PLL encoding that yields row index + raw sequence."""

    def __init__(
        self,
        df,
        sequence_column: str,
        sequence_values: Optional[Iterable[str]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sequence_column = sequence_column
        if sequence_values is None:
            self.sequence_values = [
                normalize_raw_value(self.df.at[idx, sequence_column])
                for idx in range(len(self.df))
            ]
        else:
            values = list(sequence_values)
            if len(values) != len(self.df):
                raise ValueError("sequence_values length must match df length.")
            self.sequence_values = values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "row_index": idx,
            "sequence": self.sequence_values[idx],
        }


def sequence_row_collate(batch: list[dict[str, Any]]) -> dict[str, List[Any]]:
    return {
        "row_index": [item["row_index"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
    }


def gather_sequence_values(
    local_pairs: List[tuple[int, str]],
    num_samples: int,
    accelerator,
) -> List[str]:
    sequence_values: List[Optional[str]] = [None] * num_samples

    if accelerator.num_processes <= 1 or not dist.is_initialized():
        for row_idx, value in local_pairs:
            sequence_values[int(row_idx)] = value
    else:
        gathered: List[List[tuple[int, str]]] = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, local_pairs)
        for rank_pairs in gathered:
            if not rank_pairs:
                continue
            for row_idx, value in rank_pairs:
                sequence_values[int(row_idx)] = value

    missing = [idx for idx, val in enumerate(sequence_values) if val is None]
    if missing:
        raise ValueError(f"PLL encoded outputs missing for {len(missing)} samples.")

    return [val for val in sequence_values if val is not None]
