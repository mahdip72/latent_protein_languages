from box import Box
import json
import math
import yaml
import shutil
from pathlib import Path
import datetime
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import (
    broadcast_object_list,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ParallelismConfig,
)
import torch


def suppress_inductor_autotune_logging() -> bool:
    """
    Suppress verbose torch.compile autotune logs from inductor select_algorithm.
    """
    try:
        import logging as py_logging
        import torch._inductor.config as inductor_config
        import torch._inductor.select_algorithm as select_algorithm

        # Keep this env in sync for child workers and any late config reads.
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS"] = "0"
        select_algorithm.PRINT_AUTOTUNE = False
        if hasattr(inductor_config, "autotune_num_choices_displayed"):
            inductor_config.autotune_num_choices_displayed = 0
        if hasattr(inductor_config, "max_autotune_report_choices_stats"):
            inductor_config.max_autotune_report_choices_stats = False
        # Keep cudagraph dynamic-shape behavior but suppress noisy warning spam.
        if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraph_dynamic_shape_warn_limit"):
            inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
        class _InductorAutotuneFilter(py_logging.Filter):
            def filter(self, record):
                if record.name != "torch._inductor.select_algorithm":
                    return True
                message = record.getMessage()
                return not (
                    "Runtime error during autotuning" in message
                    and (
                        "No valid triton configs" in message
                        or "out of resource: triton_mm" in message
                    )
                    and "Ignoring this choice" in message
                )

        autotune_filter = _InductorAutotuneFilter()
        select_algorithm_logger = py_logging.getLogger("torch._inductor.select_algorithm")
        select_algorithm_logger.setLevel(py_logging.CRITICAL)
        if not any(isinstance(item, _InductorAutotuneFilter) for item in select_algorithm_logger.filters):
            select_algorithm_logger.addFilter(autotune_filter)

        original_error = select_algorithm.log.error

        def filtered_error(msg, *args, **kwargs):
            try:
                rendered = msg % args if args else str(msg)
            except Exception:
                rendered = str(msg)
            if (
                "Runtime error during autotuning" in rendered
                and (
                    "No valid triton configs" in rendered
                    or "out of resource: triton_mm" in rendered
                )
                and "Ignoring this choice" in rendered
            ):
                return None
            return original_error(msg, *args, **kwargs)

        select_algorithm.log.error = filtered_error
        warnings.filterwarnings(
            "ignore",
            message=r"TypedStorage is deprecated\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Logical operators 'and' and 'or' are deprecated for non-scalar tensors; "
            r"please use '&' or '\|' instead",
            category=UserWarning,
            module=r"torch\._inductor\.runtime\.triton_helpers",
        )
        return True
    except Exception:
        return False


def configure_compile_cache_dirs(project_root: str):
    """
    Configure persistent torch.compile cache directories under the project root.
    """
    cache_root = os.path.join(project_root, "cache")
    inductor_cache_dir = os.path.join(cache_root, "torchinductor")
    triton_cache_dir = os.path.join(cache_root, "triton")

    os.makedirs(inductor_cache_dir, exist_ok=True)
    os.makedirs(triton_cache_dir, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

    return inductor_cache_dir, triton_cache_dir


def get_fsdp_config(configs):
    fsdp_cfg = getattr(configs, "fsdp", None)
    if fsdp_cfg is not None:
        return Box(
            {
                "enabled": bool(getattr(fsdp_cfg, "enabled", getattr(configs, "use_fsdp2", False))),
                "reshard_after_forward": bool(
                    getattr(
                        fsdp_cfg,
                        "reshard_after_forward",
                        getattr(configs, "fsdp2_reshard_after_forward", False),
                    )
                ),
                "parallelism": getattr(fsdp_cfg, "parallelism", getattr(configs, "parallelism", None)),
            }
        )

    # Backward-compatible fallback for older configs that still use top-level keys.
    return Box(
        {
            "enabled": bool(getattr(configs, "use_fsdp2", False)),
            "reshard_after_forward": bool(getattr(configs, "fsdp2_reshard_after_forward", False)),
            "parallelism": getattr(configs, "parallelism", None),
        }
    )


def build_fp8_ao_recipe_kwargs():
    try:
        from accelerate.utils import AORecipeKwargs
        from torchao.float8 import Float8LinearConfig
        from torchao.float8.config import CastConfig, ScalingType
    except ImportError as exc:
        raise ImportError(
            "FP8 mixed precision requires torchao and Accelerate AORecipeKwargs support. "
            "Install torchao in the active environment and use a compatible Accelerate version."
        ) from exc

    # torchao float8 uses torch._scaled_mm which requires the inner dim (K) to
    # be a multiple of 16. In training, grad_weight uses K=M (tokens), which is
    # often not divisible by 16, so we enable padding. The high-precision
    # grad-weight path is kept for parity with the tested training recipe; it is
    # harmless for inference where gradients are not computed.
    return AORecipeKwargs(
        config=Float8LinearConfig(
            cast_config_input_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
            cast_config_grad_output_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
            pad_inner_dim=True,
        )
    )


def prepare_accelerator_handlers(configs):
    """
    Build kwargs_handlers/fsdp_plugin/parallelism_config for the Accelerator.

    When FSDP is enabled in config the model is wrapped with FSDP2. Otherwise
    standard DDP is used. New configs should use the nested ``fsdp`` block, while
    older top-level ``use_fsdp2``/``parallelism`` keys remain supported.

    Returns:
        tuple: (kwargs_handlers, fsdp_plugin, parallelism_config) ready to pass
        to ``Accelerator()``.
    """
    from datetime import timedelta

    fsdp_cfg = get_fsdp_config(configs)
    use_fsdp2 = bool(getattr(fsdp_cfg, "enabled", False))

    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(minutes=configs.accelerate_timeout_minutes)),
    ]
    if configs.train_settings.mixed_precision == "fp8":
        kwargs_handlers.append(build_fp8_ao_recipe_kwargs())

    fsdp_plugin = None
    parallelism_config = None
    if use_fsdp2:
        # FSDP2 defaults to NO_WRAP, which can shard the entire model as one unit.
        # That produces very large all-gathers and unstable startup/runtime.
        # Use transformer-based wrapping by default.
        fsdp2_auto_wrap_policy = str(
            getattr(configs, "fsdp2_auto_wrap_policy", "transformer_based_wrap")
        ).strip().lower()
        valid_policies = {"size_based_wrap", "transformer_based_wrap", "no_wrap"}
        if fsdp2_auto_wrap_policy not in valid_policies:
            raise ValueError(
                f"Invalid fsdp2_auto_wrap_policy='{fsdp2_auto_wrap_policy}'. "
                f"Expected one of: {sorted(valid_policies)}"
            )

        fsdp_plugin_kwargs = dict(
            fsdp_version=2,
            reshard_after_forward=bool(getattr(fsdp_cfg, "reshard_after_forward", False)),
            state_dict_type="FULL_STATE_DICT",
        )
        if fsdp2_auto_wrap_policy != "no_wrap":
            fsdp_plugin_kwargs["auto_wrap_policy"] = fsdp2_auto_wrap_policy
            if fsdp2_auto_wrap_policy == "size_based_wrap":
                fsdp2_min_num_params = int(getattr(configs, "fsdp2_min_num_params", 10_000_000))
                fsdp_plugin_kwargs["min_num_params"] = max(1, fsdp2_min_num_params)
            elif fsdp2_auto_wrap_policy == "transformer_based_wrap":
                default_cls_names = ["FeedForward", "Attention", "FFN", "NormAttentionNorm"]
                cls_names = getattr(configs, "fsdp2_transformer_cls_names_to_wrap", default_cls_names)
                if isinstance(cls_names, str):
                    cls_names = [item.strip() for item in cls_names.split(",") if item.strip()]
                fsdp_plugin_kwargs["transformer_cls_names_to_wrap"] = list(cls_names)

        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs)
        parallelism = getattr(fsdp_cfg, "parallelism", None)
        if parallelism is not None:
            dp_shard_size = int(getattr(parallelism, "dp_shard_size", 1))
            dp_replicate_size = int(getattr(parallelism, "dp_replicate_size", 1))
            tp_size = int(getattr(parallelism, "tp_size", 1))
            cp_size = int(getattr(parallelism, "cp_size", 1))
            # Only enable ParallelismConfig when at least one dimension is > 1.
            # This avoids forcing a mesh for single-process/default runs.
            if any(size > 1 for size in (dp_shard_size, dp_replicate_size, tp_size, cp_size)):
                parallelism_config = ParallelismConfig(
                    dp_shard_size=dp_shard_size,
                    dp_replicate_size=dp_replicate_size,
                    tp_size=tp_size,
                    cp_size=cp_size,
                )
    else:
        kwargs_handlers.append(
            DistributedDataParallelKwargs(find_unused_parameters=configs.find_unused_parameters)
        )

    return kwargs_handlers, fsdp_plugin, parallelism_config


def load_yaml_box(config_path: str) -> Box:
    """
    Load YAML config file into a Box object.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Box object containing the configuration.
    """
    with open(config_path) as handle:
        data = yaml.full_load(handle)
    return Box(data)


def load_trained_run_configs(trained_dir: str, config_filename: str) -> Box:
    """
    Load the training configuration from a saved run.
    
    Args:
        trained_dir: Directory containing the saved training run.
        config_filename: Name of the config file (e.g., 'config.yaml').
        
    Returns:
        Box object containing the training configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is malformed.
    """
    config_path = os.path.join(trained_dir, config_filename)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config not found at {config_path}")

    with open(config_path) as handle:
        base_cfg = yaml.full_load(handle)
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Malformed config file: {config_path}")

    return Box(base_cfg)


def create_inference_result_dir(
    accelerator: Accelerator,
    output_base_dir: str,
    config_path: str,
) -> str:
    """
    Create timestamped result directory for inference and copy config file.
    
    This function is designed for distributed settings where only the main
    process creates the directory and the path is broadcast to all processes.
    
    Args:
        accelerator: Accelerator instance for distributed coordination.
        output_base_dir: Base directory where results should be stored.
        config_path: Path to the config file to copy into the result directory.
        
    Returns:
        Path to the created result directory (same on all processes).
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    if accelerator.is_main_process:
        result_dir = os.path.join(output_base_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy(config_path, result_dir)
        paths = [result_dir]
    else:
        paths = [None]

    broadcast_object_list(paths, from_process=0)
    return paths[0]


def flatten_records(records: Iterable[Any]) -> List[Dict[str, Any]]:
    """
    Flatten potentially nested list of records.
    
    Useful for gathering results from multiple processes where each process
    may return a list of records.
    
    Args:
        records: Iterable of records, where each record can be a dict or a list of dicts.
        
    Returns:
        Flattened list of all record dictionaries.
    """
    flattened: List[Dict[str, Any]] = []
    for entry in records:
        if isinstance(entry, list):
            flattened.extend(entry)
        else:
            flattened.append(entry)
    return flattened


def sanitize_plddt_dict(plddt_by_pid: Dict[str, float | None] | None) -> Dict[str, float | None]:
    """
    Normalize pLDDT mappings to plain floats (or None), dropping NaNs.
    """
    if not plddt_by_pid:
        return {}
    sanitized: Dict[str, float | None] = {}
    for pid, value in plddt_by_pid.items():
        if value is None:
            sanitized[str(pid)] = None
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            sanitized[str(pid)] = None
            continue
        sanitized[str(pid)] = None if math.isnan(fval) else fval
    return sanitized


def plddt_cache_dir(result_dir: str) -> str:
    """
    Resolve the cache directory used to store per-rank pLDDT summaries.
    """
    return os.path.join(result_dir, "temp_plddt_cache")


def write_plddt_cache(
    plddt_by_pid: Dict[str, float | None] | None,
    out_dir: str,
    rank: int,
    logger,
) -> str:
    """
    Persist per-rank pLDDT summaries to disk for later aggregation.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"plddt_rank_{rank}.json")
    payload = sanitize_plddt_dict(plddt_by_pid)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    logger.info("Wrote pLDDT cache: %s (%d entries)", out_path, len(payload))
    return out_path


def load_plddt_caches(
    cache_dir: str,
    num_processes: int,
    logger,
    cleanup: bool = True,
) -> Dict[str, float | None]:
    """
    Load per-rank pLDDT caches and merge into a single mapping.
    """
    merged: Dict[str, float | None] = {}
    missing = []
    for rank in range(num_processes):
        path = os.path.join(cache_dir, f"plddt_rank_{rank}.json")
        if not os.path.exists(path):
            missing.append(rank)
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load pLDDT cache %s: %s", path, exc)
            continue
        if isinstance(data, dict):
            for pid, value in data.items():
                if value is None:
                    merged[str(pid)] = None
                else:
                    try:
                        merged[str(pid)] = float(value)
                    except (TypeError, ValueError):
                        merged[str(pid)] = None
    if missing:
        logger.warning("Missing pLDDT cache files for ranks: %s", missing)
    if cleanup and os.path.isdir(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logger.info("Removed pLDDT cache dir: %s", cache_dir)
        except OSError as exc:
            logger.warning("Failed to remove pLDDT cache dir %s: %s", cache_dir, exc)
    return merged


def ca_cache_dir(result_dir: str) -> str:
    """
    Resolve the cache directory used to store per-rank CA coordinate caches.
    """
    return os.path.join(result_dir, "temp_ca_cache")


def write_ca_cache(
    ca_cache: Dict[str, Dict[int, tuple[float, float, float]]] | None,
    out_dir: str,
    rank: int,
    kind: str,
    logger,
) -> str:
    """
    Persist per-rank CA coordinate cache to disk (torch.save).
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_kind = "pred" if kind == "pred" else "true"
    out_path = os.path.join(out_dir, f"{safe_kind}_rank_{rank}.pt")
    payload = ca_cache or {}
    torch.save(payload, out_path)
    logger.info("Wrote CA cache: %s (%d entries)", out_path, len(payload))
    return out_path


def load_ca_caches(
    cache_dir: str,
    num_processes: int,
    kind: str,
    logger,
    cleanup: bool = True,
) -> Dict[str, Dict[int, tuple[float, float, float]]]:
    """
    Load per-rank CA caches and merge into a single mapping.
    """
    merged: Dict[str, Dict[int, tuple[float, float, float]]] = {}
    missing = []
    safe_kind = "pred" if kind == "pred" else "true"
    for rank in range(num_processes):
        path = os.path.join(cache_dir, f"{safe_kind}_rank_{rank}.pt")
        if not os.path.exists(path):
            missing.append(rank)
            continue
        try:
            data = torch.load(path)
        except OSError as exc:
            logger.warning("Failed to load CA cache %s: %s", path, exc)
            continue
        if isinstance(data, dict):
            for pid, coords in data.items():
                if pid not in merged:
                    merged[str(pid)] = coords
    if missing:
        logger.warning("Missing CA cache files for ranks: %s", missing)
    if cleanup and os.path.isdir(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logger.info("Removed CA cache dir: %s", cache_dir)
        except OSError as exc:
            logger.warning("Failed to remove CA cache dir %s: %s", cache_dir, exc)
    return merged


def save_tokenizer_vocab(tokenizer, output_dir: str, filename: str = "tokenizer_vocab.yaml"):
    """
    Persist tokenizer vocabulary mappings for reproducibility.
    Saves both token_to_id and id_to_token dicts into a YAML file.
    """
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, filename)
    payload = {
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
    }
    with open(vocab_path, "w") as f:
        yaml.safe_dump(payload, f)
    return vocab_path


def load_configs(config):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)

    return tree_config


def set_datasets_curriculum_epoch(epoch: int, *datasets: Any) -> None:
    """
    Set curriculum epoch on datasets with backward-compatible fallback.

    Prefers `set_curriculum_epoch` (used for condition scheduling). Falls
    back to `set_epoch` for older dataset implementations.
    """
    for dataset in datasets:
        if dataset is None:
            continue
        if hasattr(dataset, 'set_curriculum_epoch'):
            dataset.set_curriculum_epoch(epoch)
        elif hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Directory of configuration file.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path




def check_eos_and_max_length(seq_tokens, eos_token_id: int, generation_seq_len: int):
    """
    Check for EOS token and max length conditions in generated sequence.

    Args:
        seq_tokens: Tensor of generated token IDs
        eos_token_id: ID of the EOS token
        generation_seq_len: Maximum allowed generation length

    Returns:
        tuple: (has_eos, first_eos_pos, hit_max_len)
            - has_eos: bool, whether EOS token was found
            - first_eos_pos: int or None, position of first EOS token
            - hit_max_len: bool, whether sequence hit max length limit
    """
    # Find EOS token position to determine actual length
    eos_positions = (seq_tokens == eos_token_id).nonzero(as_tuple=True)[0]
    has_eos = len(eos_positions) > 0
    first_eos_pos = eos_positions[0].item() if has_eos else None
    raw_length = len(seq_tokens)

    hit_max_len = (
        (not has_eos and raw_length >= generation_seq_len) or
        (has_eos and first_eos_pos >= generation_seq_len - 1)
    )

    return has_eos, first_eos_pos, hit_max_len


def compute_decoded_content_length(decoded_sequence: str, merge_joiner: str) -> int:
    """
    Compute the content length of a decoded sequence, accounting for joiners.

    Args:
        decoded_sequence: The decoded sequence string (with special tokens already removed)
        merge_joiner: The joiner used to merge tokens (e.g., "", " ", ",")

    Returns:
        int: Number of content tokens in the sequence
    """
    if not decoded_sequence:
        return 0

    if merge_joiner == "":
        # No joiner - each character is a token
        return len(decoded_sequence)
    elif str(merge_joiner).isspace():
        # Whitespace joiner - split on whitespace
        return len(decoded_sequence.split())
    else:
        # Custom joiner - split on the joiner
        return len(decoded_sequence.split(merge_joiner))


def compute_amino_acid_sequence_entropy(sequence: str) -> float:
    """
    Compute per-sequence Shannon entropy in residue space.

    Whitespace is removed first so both compact amino-acid strings and
    whitespace-joined residue strings are handled the same way.
    """
    cleaned = "".join(str(sequence).split())
    if not cleaned:
        return 0.0

    counts: Dict[str, int] = {}
    for residue in cleaned:
        counts[residue] = counts.get(residue, 0) + 1

    length = len(cleaned)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
        if count
    )
    return 0.0 if abs(entropy) < 1e-12 else entropy


def resolve_sequence_entropy_filter_config(
    infer_cfg: Dict[str, Any],
    data_type: str,
) -> Dict[str, Any]:
    """
    Normalize optional de novo generation entropy filtering config.

    The filter is intentionally active only for direct amino-acid generation.
    PLL outputs need an additional PLL-to-amino-acid decoder before this entropy
    is meaningful, so a requested filter for PLL is reported as inactive.
    """
    cfg = infer_cfg.get("sequence_entropy_filter", {}) or {}
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    requested = bool(cfg.get("enabled", False))
    if not requested:
        return {"requested": False, "enabled": False, "min_entropy": None}

    threshold = cfg.get("min_entropy", cfg.get("threshold", None))
    if threshold is None:
        raise ValueError(
            "sequence_entropy_filter.enabled is True but min_entropy is not set"
        )

    min_entropy = float(threshold)
    return {
        "requested": True,
        "enabled": data_type == "amino_acid",
        "min_entropy": min_entropy,
    }
