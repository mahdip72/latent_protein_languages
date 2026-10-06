import json
import os

import numpy as np
import pandas as pd

MODALITY_COLUMN_CANDIDATES = {
    'amino_acid': ['Amino Acid Sequence', 'amino_acid_sequence'],
    'pll': ['indices', 'pll'],
    'structure': ['structures', 'structure'],
}

TRAIN_CSV_SHARD_COUNT = 20
TRAIN_CSV_SHARD_DIRNAME = "chunks"
TRAIN_CSV_SHARD_MANIFEST = "manifest.json"
TRAIN_CSV_SHARD_CHUNK_SIZE = 1_000_000
TRAIN_CSV_SHARD_PYARROW_BLOCK_SIZE = 64 * 1024 * 1024


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return unique items while preserving their original order."""
    return list(dict.fromkeys(items))


def _training_shard_dir(csv_file: str) -> str:
    """Resolve the reusable shard directory for a training CSV."""
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    return os.path.join(csv_dir, TRAIN_CSV_SHARD_DIRNAME)


def _training_shard_manifest_path(shard_dir: str) -> str:
    """Return the manifest path stored alongside reusable training shards."""
    return os.path.join(shard_dir, TRAIN_CSV_SHARD_MANIFEST)


def _training_shard_paths(shard_dir: str, num_shards: int) -> list[str]:
    """Return canonical shard paths for a given shard directory."""
    return [
        os.path.join(shard_dir, f"chunk_{idx}.csv")
        for idx in range(int(num_shards))
    ]


def _load_training_shard_manifest(manifest_path: str) -> dict | None:
    """Load a training shard manifest if it exists and is parseable."""
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_training_csv_chunks_pandas(
    csv_file: str,
    usecols: list[str],
    max_task_samples: int | None,
):
    """Yield preprocessing-sized CSV chunks using pandas."""
    read_kwargs = {
        "usecols": usecols,
        "low_memory": False,
        "chunksize": TRAIN_CSV_SHARD_CHUNK_SIZE,
    }
    if max_task_samples is not None:
        read_kwargs["nrows"] = int(max_task_samples)

    yield from pd.read_csv(csv_file, **read_kwargs)


def _iter_training_csv_chunks_pyarrow(
    csv_file: str,
    usecols: list[str],
    max_task_samples: int | None,
):
    """Yield preprocessing-sized CSV chunks using pyarrow when available.

    pyarrow's CSV reader streams record batches based on byte block size rather
    than a direct row-count target, so batches are accumulated until they reach
    roughly ``TRAIN_CSV_SHARD_CHUNK_SIZE`` rows before being materialized as a
    pandas dataframe for the existing shard-writing flow.
    """
    import pyarrow as pa
    import pyarrow.csv as pyarrow_csv

    read_options = pyarrow_csv.ReadOptions(
        block_size=TRAIN_CSV_SHARD_PYARROW_BLOCK_SIZE,
        use_threads=True,
    )
    convert_options = pyarrow_csv.ConvertOptions(include_columns=usecols)
    reader = pyarrow_csv.open_csv(
        csv_file,
        read_options=read_options,
        convert_options=convert_options,
    )

    pending_batches = []
    pending_rows = 0
    rows_remaining = None if max_task_samples is None else int(max_task_samples)

    for batch in reader:
        if rows_remaining is not None and rows_remaining <= 0:
            break

        if rows_remaining is not None and batch.num_rows > rows_remaining:
            batch = batch.slice(0, rows_remaining)

        if batch.num_rows == 0:
            continue

        pending_batches.append(batch)
        pending_rows += int(batch.num_rows)
        if rows_remaining is not None:
            rows_remaining -= int(batch.num_rows)

        if pending_rows >= TRAIN_CSV_SHARD_CHUNK_SIZE:
            yield pa.Table.from_batches(pending_batches).to_pandas()
            pending_batches = []
            pending_rows = 0

    if pending_batches:
        yield pa.Table.from_batches(pending_batches).to_pandas()


def iter_training_csv_chunks(
    csv_file: str,
    usecols: list[str],
    logging,
    *,
    max_task_samples: int | None = None,
):
    """Yield large preprocessing chunks, preferring pyarrow when importable."""
    try:
        import pyarrow  # noqa: F401
    except Exception:
        logging.info(
            "pyarrow not available for training shard preprocessing; using pandas chunks of %d rows.",
            TRAIN_CSV_SHARD_CHUNK_SIZE,
        )
        yield from _iter_training_csv_chunks_pandas(
            csv_file,
            usecols,
            max_task_samples,
        )
        return

    logging.info(
        "Using pyarrow for training shard preprocessing with a target batch size of %d rows.",
        TRAIN_CSV_SHARD_CHUNK_SIZE,
    )
    yield from _iter_training_csv_chunks_pyarrow(
        csv_file,
        usecols,
        max_task_samples,
    )


def _csv_identity(csv_file: str) -> dict:
    """Capture lightweight source-file identity for shard cache reuse."""
    stat_result = os.stat(csv_file)
    return {
        "source_csv": os.path.abspath(csv_file),
        "source_size": int(stat_result.st_size),
        "source_mtime_ns": int(stat_result.st_mtime_ns),
    }


def _is_training_shard_manifest_valid(
    manifest: dict | None,
    csv_file: str,
    usecols: list[str],
    num_shards: int,
    max_task_samples: int | None,
) -> bool:
    """Validate whether reusable training shards match the requested setup."""
    if not manifest:
        return False

    expected_identity = _csv_identity(csv_file)
    if manifest.get("source_csv") != expected_identity["source_csv"]:
        return False
    if int(manifest.get("source_size", -1)) != expected_identity["source_size"]:
        return False
    if int(manifest.get("source_mtime_ns", -1)) != expected_identity["source_mtime_ns"]:
        return False
    if int(manifest.get("num_shards", -1)) != int(num_shards):
        return False
    if list(manifest.get("columns", [])) != list(usecols):
        return False
    if manifest.get("max_task_samples") != max_task_samples:
        return False

    shard_paths = manifest.get("shard_paths", [])
    shard_lengths = manifest.get("shard_lengths", [])
    if len(shard_paths) != int(num_shards) or len(shard_lengths) != int(num_shards):
        return False

    for shard_path, shard_len in zip(shard_paths, shard_lengths):
        if int(shard_len) <= 0:
            continue
        if not os.path.isfile(shard_path):
            return False

    return True


def prepare_training_csv_shards(
    csv_file: str,
    modality_columns: dict,
    logging,
    *,
    num_shards: int = TRAIN_CSV_SHARD_COUNT,
    max_task_samples: int | None = None,
) -> dict:
    """Prepare reusable CSV shards for the training split and return the manifest.

    Shards are created once under ``<train_csv_dir>/chunks`` and reused on later
    runs as long as the source CSV identity, selected columns, shard count, and
    ``max_task_samples`` setting remain unchanged.

    Rows are assigned to shards in round-robin order during preprocessing so the
    initial source ordering is spread across shards without loading the full CSV
    into memory at once.
    """
    usecols = _dedupe_preserve_order(list(modality_columns.values()))
    shard_dir = _training_shard_dir(csv_file)
    manifest_path = _training_shard_manifest_path(shard_dir)

    manifest = _load_training_shard_manifest(manifest_path)
    if _is_training_shard_manifest_valid(
        manifest,
        csv_file,
        usecols,
        num_shards,
        max_task_samples,
    ):
        logging.info("Using existing training CSV shards from %s", shard_dir)
        return manifest

    os.makedirs(shard_dir, exist_ok=True)

    for shard_path in _training_shard_paths(shard_dir, num_shards):
        if os.path.exists(shard_path):
            os.remove(shard_path)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    shard_paths = _training_shard_paths(shard_dir, num_shards)
    shard_lengths = [0 for _ in range(int(num_shards))]
    header_written = [False for _ in range(int(num_shards))]

    total_rows = 0
    global_row_offset = 0
    logging.info(
        "Preparing training CSV shards in %s (%d shards, columns=%s)",
        shard_dir,
        int(num_shards),
        usecols,
    )
    for df_chunk in iter_training_csv_chunks(
        csv_file,
        usecols,
        logging,
        max_task_samples=max_task_samples,
    ):
        if df_chunk.empty:
            global_row_offset += len(df_chunk)
            continue

        row_ids = np.arange(global_row_offset, global_row_offset + len(df_chunk), dtype=np.int64)
        shard_assignments = row_ids % int(num_shards)

        for shard_idx in range(int(num_shards)):
            shard_df = df_chunk.loc[shard_assignments == shard_idx, usecols]
            if shard_df.empty:
                continue

            shard_df.to_csv(
                shard_paths[shard_idx],
                mode="a" if header_written[shard_idx] else "w",
                header=not header_written[shard_idx],
                index=False,
            )
            header_written[shard_idx] = True
            shard_lengths[shard_idx] += int(len(shard_df))

        total_rows += int(len(df_chunk))
        global_row_offset += int(len(df_chunk))

    manifest = {
        **_csv_identity(csv_file),
        "columns": usecols,
        "num_shards": int(num_shards),
        "max_task_samples": max_task_samples,
        "total_rows": int(total_rows),
        "shard_paths": shard_paths,
        "shard_lengths": shard_lengths,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logging.info(
        "Prepared training CSV shards: %d rows across %d shards in %s",
        int(total_rows),
        int(num_shards),
        shard_dir,
    )
    return manifest


def get_condition_cfg(configs, name: str) -> dict:
    model_cfg = getattr(configs, 'model', None)
    if model_cfg is None:
        return {}
    condition_cfg = getattr(model_cfg, 'condition_tokens', None)
    if condition_cfg is None:
        return {}
    cond_cfg = getattr(condition_cfg, name, None)
    if not cond_cfg:
        return {}
    if hasattr(cond_cfg, 'to_dict'):
        return cond_cfg.to_dict()
    if isinstance(cond_cfg, dict):
        return cond_cfg
    return dict(cond_cfg)


def require_bool_cfg(cond_cfg: dict, key: str, name: str) -> bool:
    if key not in cond_cfg:
        raise ValueError(f"condition_tokens.{name}.{key} is required when enabled")
    return bool(cond_cfg[key])


def select_modality_column(
    header_cols: list[str],
    modality: str,
    override: str | None = None,
) -> str:
    if override and override in header_cols:
        return override

    candidates = MODALITY_COLUMN_CANDIDATES.get(modality, [])
    for candidate in candidates:
        if candidate in header_cols:
            return candidate

    raise ValueError(
        "Required column not found for modality "
        f"'{modality}'. Looked for {candidates}; available columns: {header_cols}"
    )


def select_csv_columns(
    csv_file: str,
    data_type: str,
    main_column_name: str,
    sequence_to_structure_sequence_modality: str | None = None,
    structure_to_sequence_sequence_modality: str | None = None,
    extra_modalities: list[str] | None = None,
) -> dict:
    """Resolve CSV columns for every modality needed by a dataset.

    Uses main_column_name for the base data_type and validates all required
    modalities against the CSV header. extra_modalities forces additional
    modalities to be loaded for auxiliary mappings (e.g., PLL mode using
    amino-acid residues for missing-residue alignment).
    """
    if data_type not in MODALITY_COLUMN_CANDIDATES:
        raise ValueError(f"Unknown data_type: {data_type}")

    header_cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
    pair_modalities = [
        sequence_to_structure_sequence_modality,
        structure_to_sequence_sequence_modality,
    ]
    required_modalities = [data_type, *[m for m in pair_modalities if m]]
    if any(pair_modalities):
        required_modalities.append('structure')
    if extra_modalities:
        required_modalities.extend(extra_modalities)
    required_modalities = list(dict.fromkeys(required_modalities))

    overrides = {data_type: main_column_name}
    return {
        modality: select_modality_column(header_cols, modality, overrides.get(modality))
        for modality in required_modalities
    }


def select_columns(configs, tokenizer, logging, main_column_name: str | None = None) -> dict:
    """Select modality columns for train/validation datasets and log the choice.

    Returns a dict with the resolved data_type, main_column_name, and per-split
    modality column mappings after validating both CSV headers. Adds any extra
    modalities required for pairing guards (e.g., amino_acid for PLL mapping).
    """
    data_type = getattr(configs.model, 'data_type', 'amino_acid')
    if data_type not in MODALITY_COLUMN_CANDIDATES:
        raise ValueError(f"Unknown data_type: {data_type}")

    main_column_name = main_column_name or MODALITY_COLUMN_CANDIDATES[data_type][0]
    logging.info(f"Using data_type: {data_type}, column: '{main_column_name}'")

    extra_modalities = []
    if (
        tokenizer.sequence_to_structure_use_pll
        and tokenizer.sequence_to_structure_missing_residue_mapping
    ):
        extra_modalities.append('amino_acid')
    if (
        tokenizer.structure_to_sequence_use_pll
        and tokenizer.structure_to_sequence_missing_residue_mapping
    ):
        extra_modalities.append('amino_acid')
    extra_modalities = list(dict.fromkeys(extra_modalities))

    train_columns = select_csv_columns(
        csv_file=configs.train_settings.data_path,
        data_type=data_type,
        main_column_name=main_column_name,
        sequence_to_structure_sequence_modality=tokenizer.sequence_to_structure_sequence_modality,
        structure_to_sequence_sequence_modality=tokenizer.structure_to_sequence_sequence_modality,
        extra_modalities=extra_modalities,
    )
    val_columns = select_csv_columns(
        csv_file=configs.valid_settings.data_path,
        data_type=data_type,
        main_column_name=main_column_name,
        sequence_to_structure_sequence_modality=tokenizer.sequence_to_structure_sequence_modality,
        structure_to_sequence_sequence_modality=tokenizer.structure_to_sequence_sequence_modality,
        extra_modalities=extra_modalities,
    )

    return {
        'data_type': data_type,
        'main_column_name': main_column_name,
        'train': train_columns,
        'val': val_columns,
    }


def init_structure_condition_settings(dataset, tokenizer, configs, mode: str) -> None:
    """Initialize structure pairing settings on the dataset from tokenizer and config."""
    sequence_to_structure_enabled = any(
        c.name == 'sequence_to_structure' and c.enabled
        for c in getattr(tokenizer, 'conditions', [])
    )
    structure_to_sequence_enabled = any(
        c.name == 'structure_to_sequence' and c.enabled
        for c in getattr(tokenizer, 'conditions', [])
    )
    structure_pair_enabled = sequence_to_structure_enabled or structure_to_sequence_enabled

    sequence_to_structure_cfg = get_condition_cfg(configs, 'sequence_to_structure')
    structure_to_sequence_cfg = get_condition_cfg(configs, 'structure_to_sequence')
    seq2struct_context_cfg = sequence_to_structure_cfg.get('protein_encoder_context', {})
    sequence_to_structure_use_protein_encoder_context = bool(
        seq2struct_context_cfg.get('enable', seq2struct_context_cfg.get('enabled', False))
    )

    sequence_to_structure_use_pll = None
    structure_to_sequence_use_pll = None
    if sequence_to_structure_enabled:
        if sequence_to_structure_use_protein_encoder_context:
            sequence_to_structure_use_pll = False
        else:
            sequence_to_structure_use_pll = require_bool_cfg(
                sequence_to_structure_cfg, 'use_pll', 'sequence_to_structure'
            )
    if structure_to_sequence_enabled:
        structure_to_sequence_use_pll = require_bool_cfg(
            structure_to_sequence_cfg, 'use_pll', 'structure_to_sequence'
        )

    force_sequence_to_structure_in_val = bool(
        mode == 'val'
        and sequence_to_structure_cfg.get('include_structure_in_validation', False)
    )
    force_structure_to_sequence_in_val = bool(
        mode == 'val'
        and structure_to_sequence_cfg.get('include_sequence_in_validation', False)
    )
    force_structure_in_val = bool(
        force_sequence_to_structure_in_val or force_structure_to_sequence_in_val
    )

    mask_non_structure_loss = {
        'sequence_to_structure': bool(
            sequence_to_structure_cfg.get('mask_non_structure_loss', False)
        ),
        'structure_to_sequence': bool(
            structure_to_sequence_cfg.get('mask_non_sequence_loss', False)
        ),
    }

    sequence_to_structure_sequence_modality = (
        'pll' if sequence_to_structure_use_pll else 'amino_acid'
        if sequence_to_structure_enabled
        else None
    )
    structure_to_sequence_sequence_modality = (
        'pll' if structure_to_sequence_use_pll else 'amino_acid'
        if structure_to_sequence_enabled
        else None
    )

    dataset.sequence_to_structure_enabled = sequence_to_structure_enabled
    dataset.structure_to_sequence_enabled = structure_to_sequence_enabled
    dataset.structure_pair_enabled = structure_pair_enabled
    dataset.sequence_to_structure_cfg = sequence_to_structure_cfg
    dataset.structure_to_sequence_cfg = structure_to_sequence_cfg
    dataset.sequence_to_structure_use_pll = sequence_to_structure_use_pll
    dataset.sequence_to_structure_use_protein_encoder_context = sequence_to_structure_use_protein_encoder_context
    dataset.sequence_to_structure_protein_encoder_context_cfg = seq2struct_context_cfg
    dataset.structure_to_sequence_use_pll = structure_to_sequence_use_pll
    dataset.force_sequence_to_structure_in_val = force_sequence_to_structure_in_val
    dataset.force_structure_to_sequence_in_val = force_structure_to_sequence_in_val
    dataset.force_structure_in_val = force_structure_in_val
    dataset.mask_non_structure_loss = mask_non_structure_loss
    dataset.sequence_to_structure_sequence_modality = sequence_to_structure_sequence_modality
    dataset.structure_to_sequence_sequence_modality = structure_to_sequence_sequence_modality


def extract_condition_config(model_cfg) -> dict | None:
    """
    Extract condition_tokens config from model config.
    Handles Box objects and converts to plain dict.
    """
    if not hasattr(model_cfg, 'condition_tokens'):
        return None

    config = model_cfg.condition_tokens
    if not config:
        return None

    # Convert Box/mapping to dict
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    if not isinstance(config, dict):
        return dict(config)
    return config


def log_conditioning_status(tokenizer, max_len: int, logging) -> None:
    """Log which conditions are enabled."""
    if not getattr(tokenizer, 'conditioning_enabled', False):
        logging.info("Conditioning DISABLED (standard mode)")
        logging.info(f"Tokenizer vocabulary size: {tokenizer.tokenizer_vocab_size}")
        return

    logging.info("Conditioning ENABLED:")

    for cond in getattr(tokenizer, 'conditions', []):
        if cond.enabled:
            logging.info(f"  - {cond.name}: prob={cond.prob}")

    logging.info("  - Special markers: <BOP>, <EOP>")
    logging.info(f"Tokenizer vocabulary size: {tokenizer.tokenizer_vocab_size}")
