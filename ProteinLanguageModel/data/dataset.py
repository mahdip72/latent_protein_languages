from bisect import bisect_right

from torch.utils.data import Dataset, DataLoader
import inspect
import pandas as pd
import random
import torch
from data.tokenizer import ProteinTokenizer
from data.utils import (
    select_columns,
    init_structure_condition_settings,
    extract_condition_config,
    log_conditioning_status,
    prepare_training_csv_shards,
)


def _read_csv_supports_dtype_backend() -> bool:
    """Return True if pandas.read_csv supports dtype_backend."""
    try:
        return "dtype_backend" in inspect.signature(pd.read_csv).parameters
    except (TypeError, ValueError):
        return False


def _maybe_enable_pyarrow(read_kwargs: dict) -> tuple[dict, bool]:
    """Enable pyarrow CSV engine when available; otherwise return kwargs unchanged."""
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return read_kwargs, False

    updated = dict(read_kwargs)
    updated["engine"] = "pyarrow"
    if _read_csv_supports_dtype_backend():
        updated["dtype_backend"] = "pyarrow"
    updated.pop("low_memory", None)
    return updated, True


class _ShardBackedColumn:
    """Sequence-like proxy that resolves values from the currently active shard."""

    def __init__(self, dataset, modality: str):
        self.dataset = dataset
        self.modality = modality

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset._get_shard_value(self.modality, idx)


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein sequences.
    
    Loads sequences from a CSV file and tokenizes them using the provided tokenizer.
    Supports both standard and conditioned tokenization modes with per-sample
    modality dicts (e.g., {'amino_acid': ..., 'structure': ...}).
    
    Args:
        csv_file: Path to CSV file with protein sequences
        tokenizer: ProteinTokenizer instance
        configs: Configuration object with model.max_len and train_settings
        main_column_name: Primary data_type column name selected upstream
        mode: 'train' or 'val' (affects sample limiting)
        data_type: 'amino_acid', 'pll', or 'structure'
        modality_columns: Required mapping of modality -> CSV column name (from select_columns)
    """
    
    def __init__(
        self,
        csv_file: str,
        tokenizer: ProteinTokenizer,
        configs,
        main_column_name: str = 'Amino Acid Sequence',
        mode: str = 'val',
        data_type: str = 'amino_acid',
        modality_columns: dict | None = None,
    ):
        init_structure_condition_settings(self, tokenizer, configs, mode)
        if modality_columns is None:
            raise ValueError("modality_columns must be provided; call select_columns first")
        self.modality_columns = modality_columns
        self.main_column_name = main_column_name
        self._columns_by_modality: dict[str, list] = {m: [] for m in self.modality_columns}
        self._num_samples: int = 0

        # Load samples
        usecols = list(dict.fromkeys(self.modality_columns.values()))
        read_kwargs = {'usecols': usecols, 'low_memory': False}
        if mode == 'train':
            read_kwargs['nrows'] = configs.train_settings.max_task_samples

        chunksize = 0
        if mode == 'train':
            # Hardcode chunk size to 1/10th of the configured sample cap to reduce peak RAM
            # during dataset initialization.
            nrows = int(read_kwargs.get('nrows', 0) or 0)
            if nrows > 0:
                chunksize = max(1, nrows // 10)

        if chunksize > 0:
            # NOTE: pandas' pyarrow CSV engine does not support `chunksize`, so chunked
            # loading uses the default (C) engine.
            print(f"Using chunked CSV loading (chunksize={chunksize}).")
            chunk_kwargs = dict(read_kwargs)
            chunk_kwargs['chunksize'] = chunksize
            for df_chunk in pd.read_csv(csv_file, **chunk_kwargs):
                self._extend_columns(df_chunk)
                del df_chunk
        else:
            base_read_kwargs = dict(read_kwargs)
            read_kwargs, pyarrow_enabled = _maybe_enable_pyarrow(read_kwargs)
            try:
                df = pd.read_csv(csv_file, **read_kwargs)
                if pyarrow_enabled:
                    print("Using pyarrow engine for CSV loading.")
            except (TypeError, ValueError):
                if pyarrow_enabled:
                    df = pd.read_csv(csv_file, **base_read_kwargs)
                else:
                    raise
            self._extend_columns(df)
            del df
        self._finalize_columns()

        self._initialize_runtime_state(tokenizer, configs, mode, data_type)

    def _initialize_runtime_state(self, tokenizer, configs, mode: str, data_type: str) -> None:
        """Initialize runtime attributes shared by eager and shard-backed datasets."""
        self.tokenizer = tokenizer
        self.max_len = configs.model.max_len
        self.data_type = data_type
        # For validation, keep conditioning enabled but force no condition to fire
        self.force_no_condition = (mode == 'val')
        self.is_training = (mode == 'train')
        self.no_condition_overrides = dict(self._build_no_condition_overrides())
        self._e1_vocab = None
        
        # Epoch-based conditioning control (per-condition start epochs).
        # Keep curriculum epoch separate from sampler epoch because wrappers
        # like Accelerate may call `set_epoch(0)` for shuffling.
        self.current_epoch = 1
        self.sampler_epoch = 0
        self.condition_start_epochs = self._build_condition_start_epochs(configs)

    def _extend_columns(self, df: pd.DataFrame) -> None:
        """Append a dataframe chunk into column storage (no per-row dict materialization)."""
        for modality, column in self.modality_columns.items():
            # tolist() keeps references to existing Python objects (no deep copy of strings)
            # and is significantly cheaper than building list[dict] of rows.
            self._columns_by_modality[modality].extend(df[column].tolist())

    def _finalize_columns(self) -> None:
        lengths = {m: len(v) for m, v in self._columns_by_modality.items()}
        if not lengths:
            self._num_samples = 0
            return
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(f"Dataset column lengths mismatch: {lengths}")
        self._num_samples = next(iter(unique))

    def _get_e1_vocab(self) -> dict[str, int]:
        if self._e1_vocab is None:
            try:
                from E1.tokenizer import get_tokenizer
            except ImportError as exc:
                raise ImportError(
                    "protein_encoder_context with model_type='e1' requires the E1 package."
                ) from exc
            self._e1_vocab = get_tokenizer().get_vocab()
        return self._e1_vocab

    def _build_e1_context_inputs(self, protein_sequence: str) -> dict[str, torch.Tensor]:
        seq = str(protein_sequence).upper()
        if "," in seq:
            raise ValueError(
                "E1 context integration expects single sequences only; found a comma-separated entry."
            )

        seq = seq[: self.max_len]
        seq_len = len(seq)
        vocab = self._get_e1_vocab()

        pad = vocab["<pad>"]
        bos = vocab["<bos>"]
        eos = vocab["<eos>"]
        one = vocab["1"]
        two = vocab["2"]
        x_id = vocab.get("X", pad)

        total_len = self.max_len + 4
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

        residue_attention_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if seq_len > 0:
            residue_attention_mask[:seq_len] = True

        return {
            'protein_encoder_context_input_ids': input_ids,
            'protein_encoder_context_within_seq_position_ids': within_seq_position_ids,
            'protein_encoder_context_global_position_ids': global_position_ids,
            'protein_encoder_context_sequence_ids': sequence_ids,
            'protein_encoder_context_attention_mask': residue_attention_mask,
        }

    def set_epoch(self, epoch: int) -> None:
        """Update sampler epoch (typically 0-indexed from distributed wrappers)."""
        self.sampler_epoch = int(epoch)

    def set_curriculum_epoch(self, epoch: int) -> None:
        """Update curriculum epoch used by condition start_epoch gating (1-indexed)."""
        self.current_epoch = max(1, int(epoch))
    
    def _build_condition_start_epochs(self, configs) -> dict:
        """Build a mapping of condition_name -> start_epoch from config."""
        start_epochs = {}
        condition_tokens = getattr(getattr(configs, 'model', None), 'condition_tokens', None)
        if condition_tokens is None:
            return start_epochs
        
        for cond in getattr(self.tokenizer, 'conditions', []):
            cond_cfg = getattr(condition_tokens, cond.name, None)
            if cond_cfg is not None:
                # Get start_epoch from condition config, default to 1
                start_epoch = getattr(cond_cfg, 'start_epoch', 1)
                start_epochs[cond.name] = int(start_epoch) if start_epoch else 1
            else:
                start_epochs[cond.name] = 1
        
        return start_epochs
    
    def _build_no_condition_overrides(self) -> dict:
        """Precompute per-condition overrides to disable all conditions."""
        if not (self.force_no_condition and getattr(self.tokenizer, 'conditioning_enabled', False)):
            return {}
        return {f'apply_{cond.name}': False for cond in getattr(self.tokenizer, 'conditions', [])}
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        sample = {m: self._columns_by_modality[m][idx] for m in self._columns_by_modality}
        sequence = sample.get(self.data_type)

        # Build per-condition overrides based on current epoch
        # Each condition can have its own start_epoch for fine-grained curriculum control
        if self.is_training:
            condition_overrides = {}
            for cond in getattr(self.tokenizer, 'conditions', []):
                start_epoch = self.condition_start_epochs.get(cond.name, 1)
                if self.current_epoch < start_epoch:
                    # Force this specific condition off
                    condition_overrides[f'apply_{cond.name}'] = False
            # Apply validation overrides on top (for structure pairing logic)
            condition_overrides = self._apply_structure_pair_validation_override(condition_overrides)
        else:
            # Validation: use standard override logic
            condition_overrides = self._apply_structure_pair_validation_override(
                dict(self.no_condition_overrides)
            )
        
        encoded = self.tokenizer.encode(
            sample,
            data_type=self.data_type,
            max_length=self.max_len,
            mask_non_structure_loss=self.mask_non_structure_loss if self.structure_pair_enabled else False,
            is_training=self.is_training,
            **condition_overrides,
        )
        
        output = {
            'input_ids': encoded['input_ids'],
            'target_ids': encoded['target_ids'],
            'mask': encoded['mask'],
            'tokenized_sequence': encoded['tokenized_sequence'],
            'non_pad_length': encoded['non_pad_length'],
            'unmasked_token_count': encoded['unmasked_token_count'],
            'original_sequence': sequence,
            **{k: v for k, v in encoded.items() if k.startswith('is_')},
        }

        if getattr(self, 'sequence_to_structure_use_protein_encoder_context', False):
            amino_value = sample.get('amino_acid')
            if amino_value is None:
                raise ValueError(
                    "protein_encoder_context requires an amino_acid column in the dataset sample."
                )
            output.update(self._build_e1_context_inputs(str(amino_value)))

        return output

    def _apply_structure_pair_validation_override(self, condition_overrides: dict) -> dict:
        """Force sequence/structure pairing during validation when configured.

        If both pairing conditions are enabled for validation, randomly select
        one (0.5 each) to avoid mixing them within the same sample.
        """
        if not (self.force_structure_in_val and self.structure_pair_enabled):
            return condition_overrides

        eligible = []
        if self.sequence_to_structure_enabled and self.force_sequence_to_structure_in_val:
            eligible.append('sequence_to_structure')
        if self.structure_to_sequence_enabled and self.force_structure_to_sequence_in_val:
            eligible.append('structure_to_sequence')
        if not eligible:
            return condition_overrides

        if len(eligible) == 2:
            if random.random() < 0.5:
                condition_overrides['apply_sequence_to_structure'] = True
            else:
                condition_overrides['apply_structure_to_sequence'] = True
        elif eligible[0] == 'sequence_to_structure':
            condition_overrides['apply_sequence_to_structure'] = True
        else:
            condition_overrides['apply_structure_to_sequence'] = True

        return condition_overrides


class ShardedProteinDataset(ProteinDataset):
    """Training-only dataset that loads one reusable CSV shard into memory at a time.

    Tokenization and sample construction are intentionally inherited from
    :class:`ProteinDataset`. The only behavior change is how raw rows are sourced:
    reusable offline CSV shards are loaded lazily, shuffled per epoch, and freed
    before moving to the next shard.
    """

    def __init__(
        self,
        shard_manifest: dict,
        tokenizer: ProteinTokenizer,
        configs,
        main_column_name: str = 'Amino Acid Sequence',
        mode: str = 'train',
        data_type: str = 'amino_acid',
        modality_columns: dict | None = None,
    ):
        if mode != 'train':
            raise ValueError("ShardedProteinDataset is intended for training mode only.")

        init_structure_condition_settings(self, tokenizer, configs, mode)
        if modality_columns is None:
            raise ValueError("modality_columns must be provided; call select_columns first")

        self.modality_columns = modality_columns
        self.main_column_name = main_column_name
        self._num_samples = int(shard_manifest.get('total_rows', 0))
        self._shard_paths = list(shard_manifest.get('shard_paths', []))
        self._shard_lengths = [int(length) for length in shard_manifest.get('shard_lengths', [])]
        self._shuffle_seed_base = int(getattr(configs, 'fix_seed', 0) or 0)
        self._csv_usecols = list(dict.fromkeys(self.modality_columns.values()))

        self._active_shard_position: int | None = None
        self._active_shard_id: int | None = None
        self._active_shard_columns: dict[str, list] = {}
        self._active_row_order: list[int] = []

        self._nonempty_shard_ids = [
            shard_id
            for shard_id, shard_len in enumerate(self._shard_lengths)
            if int(shard_len) > 0
        ]
        self._epoch_shard_order = list(self._nonempty_shard_ids)
        self._epoch_shard_cumulative: list[int] = []

        self._columns_by_modality = {
            modality: _ShardBackedColumn(self, modality)
            for modality in self.modality_columns
        }

        self._initialize_runtime_state(tokenizer, configs, mode, data_type)
        self._rebuild_epoch_shard_order()

    def _rebuild_epoch_shard_order(self) -> None:
        """Shuffle shard order for the current epoch and reset the active shard cache."""
        shard_order = list(self._nonempty_shard_ids)
        rng = random.Random(self._shuffle_seed_base + int(self.current_epoch))
        rng.shuffle(shard_order)

        cumulative = []
        running = 0
        for shard_id in shard_order:
            running += int(self._shard_lengths[shard_id])
            cumulative.append(running)

        self._epoch_shard_order = shard_order
        self._epoch_shard_cumulative = cumulative
        self._active_shard_position = None
        self._active_shard_id = None
        self._active_shard_columns = {}
        self._active_row_order = []

    def _load_active_shard(self, shard_position: int) -> None:
        """Load the requested shard into memory and shuffle its row order for this epoch."""
        shard_id = self._epoch_shard_order[shard_position]
        if self._active_shard_position == shard_position and self._active_shard_id == shard_id:
            return

        shard_path = self._shard_paths[shard_id]
        df = pd.read_csv(shard_path, usecols=self._csv_usecols, low_memory=False)
        shard_columns = {
            modality: df[column].tolist()
            for modality, column in self.modality_columns.items()
        }
        del df

        row_order = list(range(int(self._shard_lengths[shard_id])))
        rng = random.Random(
            self._shuffle_seed_base
            + int(self.current_epoch) * 1009
            + int(shard_id)
        )
        rng.shuffle(row_order)

        self._active_shard_position = shard_position
        self._active_shard_id = shard_id
        self._active_shard_columns = shard_columns
        self._active_row_order = row_order

    def _resolve_shard_position(self, idx: int) -> tuple[int, int]:
        """Map a dataset index onto the currently shuffled shard order."""
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._num_samples}")

        shard_position = bisect_right(self._epoch_shard_cumulative, idx)
        shard_start = 0 if shard_position == 0 else self._epoch_shard_cumulative[shard_position - 1]
        local_idx = idx - shard_start
        return shard_position, local_idx

    def _get_shard_value(self, modality: str, idx: int):
        """Resolve a modality value from the active shard cache for a dataset index."""
        shard_position, local_idx = self._resolve_shard_position(int(idx))
        self._load_active_shard(shard_position)
        row_idx = self._active_row_order[local_idx]
        return self._active_shard_columns[modality][row_idx]

    def set_curriculum_epoch(self, epoch: int) -> None:
        """Update conditioning epoch and rebuild the shard order for the new epoch."""
        super().set_curriculum_epoch(epoch)
        self._rebuild_epoch_shard_order()


def prepare_dataloaders(configs, logging, accelerator=None):
    """
    Prepare training and validation dataloaders.
    
    Args:
        configs: Configuration object with:
            - model.data_type: 'amino_acid', 'pll', or 'structure'
            - model.max_len: Maximum sequence length
            - model.condition_tokens (optional): Conditioning config
            - train_settings.*: Training dataloader settings
            - valid_settings.*: Validation dataloader settings
        logging: Logger instance (used to report selected columns)
        accelerator: Optional Accelerator used to coordinate one-time shard
            preprocessing across ranks before the training dataset is built.
    
    Returns:
        Tuple of (train_loader, val_loader, tokenizer).

    Notes:
        The training split is loaded through reusable CSV shards created once
        under the training data directory. Validation keeps the original eager
        in-memory dataset path because it is much smaller.
    """
    # Determine data type
    data_type = getattr(configs.model, 'data_type', 'amino_acid')
    
    # Create tokenizer
    condition_config = extract_condition_config(configs.model)
    max_len = configs.model.max_len
    
    tokenizer = ProteinTokenizer(
        data_type=data_type,
        condition_config=condition_config,
        max_len=max_len,
        vocab_path=getattr(configs.model, 'tokenizer_vocab_path', None),
    )
    
    # Log conditioning status
    log_conditioning_status(tokenizer, max_len, logging)
    
    column_selection = select_columns(configs, tokenizer, logging)
    main_column_name = column_selection['main_column_name']
    train_modality_columns = column_selection['train']
    val_modality_columns = column_selection['val']

    train_shard_manifest = None
    max_task_samples = getattr(configs.train_settings, 'max_task_samples', None)
    if accelerator is None or accelerator.is_main_process:
        train_shard_manifest = prepare_training_csv_shards(
            configs.train_settings.data_path,
            train_modality_columns,
            logging,
            max_task_samples=max_task_samples,
        )
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is not None and not accelerator.is_main_process:
        train_shard_manifest = prepare_training_csv_shards(
            configs.train_settings.data_path,
            train_modality_columns,
            logging,
            max_task_samples=max_task_samples,
        )

    # Create datasets
    train_dataset = ShardedProteinDataset(
        shard_manifest=train_shard_manifest,
        tokenizer=tokenizer,
        configs=configs,
        main_column_name=main_column_name,
        mode='train',
        data_type=data_type,
        modality_columns=train_modality_columns,
    )
    val_dataset = ProteinDataset(
        csv_file=configs.valid_settings.data_path,
        tokenizer=tokenizer,
        configs=configs,
        main_column_name=main_column_name,
        mode='val',
        data_type=data_type,
        modality_columns=val_modality_columns,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        num_workers=configs.train_settings.num_workers,
        pin_memory=configs.train_settings.pin_memory,
        persistent_workers=configs.train_settings.persistent_workers,
        shuffle=False,
        drop_last=configs.train_settings.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.valid_settings.batch_size,
        num_workers=configs.valid_settings.num_workers,
        pin_memory=configs.valid_settings.pin_memory,
        persistent_workers=configs.valid_settings.persistent_workers,
        shuffle=configs.valid_settings.shuffle,
        drop_last=configs.valid_settings.drop_last,
    )
    
    logging.info(
        "Dataloaders prepared successfully. "
        "Training shard shuffle is dataset-driven; DataLoader shuffle is disabled for train."
    )
    return train_loader, val_loader, tokenizer


# =============================================================================
# Tests (run with: python -m data.dataset from project root)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Dataset Module Tests")
    print("=" * 60)
    
    # Test basic tokenizer
    tok = ProteinTokenizer('amino_acid')
    enc = tok.encode('MDEAA', data_type='amino_acid', max_length=10)
    
    print(f"\n1. Standard mode:")
    print(f"   Input: {[tok.id_to_token[i] for i in enc['input_ids'].tolist()]}")
    assert tok.eos_token_id not in enc['input_ids'].tolist()
    print("   ✓ PASS")
    
    # Test with both conditions
    cfg = {'length': {'enabled': True, 'probability': 1.0}, 'c2n': {'enabled': True, 'probability': 1.0}}
    tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
    enc = tok.encode('MDEAA', data_type='amino_acid', max_length=20, apply_length=True, apply_c2n=True)
    
    tokens = [tok.id_to_token[i] for i in enc['input_ids'].tolist()]
    print(f"\n2. Both conditions (C2N + Length):")
    print(f"   Input: {tokens[:12]}")
    assert tokens[:4] == ['<BOS>', '<C2N>', '<LEN_5>', '<BOP>']
    print("   ✓ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nTo test with config: python -c \"...")
    print("See test commands in docstring or run from project root.")
