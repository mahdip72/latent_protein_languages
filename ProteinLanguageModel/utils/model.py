import os
from collections import OrderedDict
import torch
from typing import Optional, Tuple
from accelerate import Accelerator
import yaml


def _find_tokenizer_vocab_path(resume_path: str, resume_cfg) -> Optional[str]:
    candidates = []
    if resume_cfg is not None:
        explicit_path = getattr(resume_cfg, 'tokenizer_vocab_path', None)
        if explicit_path:
            candidates.append(explicit_path)

    if resume_path:
        if os.path.isdir(resume_path):
            candidates.append(os.path.join(resume_path, 'tokenizer_vocab.yaml'))
            candidates.append(os.path.join(os.path.dirname(resume_path), 'tokenizer_vocab.yaml'))
        else:
            resume_dir = os.path.dirname(resume_path)
            candidates.append(os.path.join(resume_dir, 'tokenizer_vocab.yaml'))
            candidates.append(os.path.join(os.path.dirname(resume_dir), 'tokenizer_vocab.yaml'))

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def _load_tokenizer_vocab(vocab_path: str) -> Optional[dict]:
    with open(vocab_path, 'r') as handle:
        payload = yaml.safe_load(handle) or {}

    token_to_id = payload.get('token_to_id')
    if token_to_id:
        return {str(token): int(idx) for token, idx in token_to_id.items()}

    id_to_token = payload.get('id_to_token')
    if id_to_token:
        return {str(token): int(idx) for idx, token in id_to_token.items()}

    return None


def _remap_embedding_weights(model_state, current_state, old_token_to_id, new_token_to_id, logging):
    emb_keys = [k for k in current_state if 'token_emb' in k and k.endswith('weight')]
    if not emb_keys:
        logging.info('No token embedding weights found for remapping.')
        return model_state

    shared_tokens = set(old_token_to_id) & set(new_token_to_id)
    if not shared_tokens:
        logging.info('No shared tokens between checkpoint and current tokenizer vocab.')
        return model_state

    for emb_key in emb_keys:
        if emb_key not in model_state:
            continue
        old_weight = model_state[emb_key]
        new_weight = current_state[emb_key].clone()

        if old_weight.ndim != 2 or new_weight.ndim != 2:
            logging.info('Skipping remap for %s due to unexpected tensor rank.', emb_key)
            continue
        if old_weight.shape[1] != new_weight.shape[1]:
            logging.info('Skipping remap for %s due to embedding dim mismatch.', emb_key)
            continue

        copied = 0
        for token in shared_tokens:
            old_idx = old_token_to_id.get(token)
            new_idx = new_token_to_id.get(token)
            if old_idx is None or new_idx is None:
                continue
            if old_idx >= old_weight.shape[0] or new_idx >= new_weight.shape[0]:
                continue
            new_weight[new_idx] = old_weight[old_idx].to(new_weight.dtype)
            copied += 1

        model_state[emb_key] = new_weight
        logging.info(
            'Remapped %d/%d tokens into %s.',
            copied,
            len(new_token_to_id),
            emb_key,
        )

    return model_state


def _drop_output_projection_weights(model_state, logging):
    dropped = []
    for key in list(model_state.keys()):
        if 'to_logits' in key or 'projector' in key:
            dropped.append(key)
            model_state.pop(key, None)
    if dropped:
        logging.info('Skipping output projection weights: %s', dropped)
    return model_state


def clean_state_dict_keys(state_dict):
    """
    Remove '_orig_mod.' prefix from state dict keys.
    
    This prefix is added by torch.compile and needs to be removed when
    loading checkpoints from compiled models into non-compiled models
    or vice versa.
    
    Args:
        state_dict: Model state dictionary, possibly with '_orig_mod.' prefixes.
        
    Returns:
        Cleaned state dictionary with prefixes removed, or None if input is None.
    """
    if state_dict is None:
        return None
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        cleaned[clean_key] = value
    return cleaned


# Keep the old name for backward compatibility
def _clean_state_dict_keys(state_dict):
    if state_dict is None:
        return None
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        cleaned[clean_key] = value
    return cleaned


def save_checkpoint(epoch: int, model_path: str, net, optimizer, accelerator: Accelerator, **kwargs):
    """
    Save training checkpoint to disk.

    Saves the current epoch number, model state dict, optimizer state dict,
    and training step count to the given file path. The model is unwrapped
    via the provided Accelerator.

    Args:
        epoch (int): Current epoch number.
        model_path (str): File path for saving the checkpoint.
        net (torch.nn.Module): Model to save (wrapped by Accelerator).
        optimizer: Optimizer instance.
        accelerator (Accelerator): Hugging Face Accelerator used to unwrap the model.
        **kwargs: Additional checkpoint data. Expected keys:
            - global_step (int): Total number of training iterations completed.
            - scheduler (optional): Scheduler instance to serialize.

    Returns:
        None
    """

    # FSDP state dict gathering uses collectives. All ranks must call
    # `accelerator.get_state_dict()` in the same order or the job can deadlock.
    is_fsdp = bool(getattr(getattr(accelerator, "state", None), "fsdp_plugin", None))

    # Non-FSDP: only rank0 writes/checkpoints to avoid duplicating large CPU work.
    # FSDP: all ranks must participate in state-dict collection, but only rank0 writes.
    if not is_fsdp and not accelerator.is_main_process:
        return

    with torch.no_grad():
        # Build checkpoint dictionary using a robust path across accelerate versions.
        # For FSDP, prefer `accelerator.get_state_dict()` (handles rank0_only gathering).
        if is_fsdp:
            model_state = accelerator.get_state_dict(net)
        else:
            try:
                model_state = accelerator.get_state_dict(net)
            except Exception:
                try:
                    base_model = accelerator.unwrap_model(net, keep_torch_compile=True)
                    base_model = getattr(base_model, '_orig_mod', base_model)
                    model_state = base_model.state_dict()
                except Exception:
                    inner = getattr(net, 'module', net)
                    inner = getattr(inner, '_orig_mod', inner)
                    model_state = inner.state_dict()

    model_state = _clean_state_dict_keys(model_state)

    # In FSDP mode, non-main ranks must exit after participating in state-dict collection.
    if is_fsdp and not accelerator.is_main_process:
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': kwargs['global_step'],
    }

    scheduler = kwargs.get('scheduler')
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, model_path)


def save_step_checkpoint_if_needed(
    epoch,
    checkpoint_path,
    global_step,
    net,
    optimizer,
    scheduler,
    accelerator,
    logging,
    configs,
):
    """Persist a checkpoint whenever the run crosses the configured step cadence.

    Mirrors the epoch-based checkpoint helper but keys off ``checkpoint_every_steps`` in the
    config. All processes synchronize before and after the optional save so the main rank is the
    only writer while the remaining workers wait for confirmation.

    Args:
        epoch (int): Current epoch number.
        checkpoint_path (str): Directory where checkpoints should be written.
        global_step (int): Total optimizer steps completed so far.
        net (torch.nn.Module): Model (possibly wrapped by ``accelerator``) to serialize.
        optimizer (torch.optim.Optimizer): Optimizer whose state should be checkpointed.
        scheduler: Optional scheduler whose state should be checkpointed.
        accelerator (Accelerator): Orchestrator providing distributed utilities.
        logging: Logger with an ``info`` method for status messages.
        configs: Configuration object exposing ``checkpoint_every_steps``.
    """

    if configs.checkpoint_every_steps <= 0:
        return

    if global_step % configs.checkpoint_every_steps != 0 or global_step == 0:
        return

    accelerator.wait_for_everyone()
    model_path = os.path.join(checkpoint_path, f'steps_{global_step}.pth')
    save_checkpoint(
        epoch,
        model_path,
        net,
        optimizer,
        accelerator,
        global_step=global_step,
        scheduler=scheduler,
    )
    if accelerator.is_main_process:
        logging.info(
            f'at step {global_step:,}, checkpoint saved in {model_path}'
        )
    accelerator.wait_for_everyone()


def load_optimizer_state(optimizer, optimizer_state, logging, param_name_map=None):
    """Best-effort optimizer state loading with shape guards and logging."""
    if optimizer_state is None:
        logging.info("No optimizer state found in checkpoint; starting fresh optimizer.")
        return False

    state_groups = optimizer_state.get('param_groups', [])
    state = optimizer_state.get('state', {})
    if not state_groups or state is None:
        logging.warning("Checkpoint optimizer state missing param_groups/state; starting fresh optimizer.")
        return False

    current_state = optimizer.state_dict()
    current_groups = current_state.get('param_groups', [])
    if not current_groups:
        logging.warning("Current optimizer has no param_groups; skipping optimizer resume.")
        return False

    param_ids_flat = [
        param_id
        for group in current_groups
        for param_id in group.get('params', [])
    ]
    params_flat = [
        param
        for group in optimizer.param_groups
        for param in group.get('params', [])
    ]
    if len(param_ids_flat) != len(params_flat):
        logging.warning(
            "Optimizer param mapping size mismatch: state_dict has %d params, optimizer has %d params.",
            len(param_ids_flat),
            len(params_flat),
        )
    param_id_to_param = dict(zip(param_ids_flat, params_flat))

    loaded_state = {
        'state': {},
        'param_groups': current_groups,
    }

    total_params = sum(len(group.get('params', [])) for group in current_groups)
    loaded_params = 0
    missing_params = 0
    shape_mismatch = 0
    extra_ckpt_groups = max(0, len(state_groups) - len(current_groups))
    extra_ckpt_params = 0
    missing_examples = []
    mismatch_examples = []
    max_examples = 10

    base_optimizer = optimizer
    visited = set()
    while hasattr(base_optimizer, "optimizer") and id(base_optimizer) not in visited:
        visited.add(id(base_optimizer))
        base_optimizer = base_optimizer.optimizer

    optimizer_module = base_optimizer.__class__.__module__
    optimizer_name = base_optimizer.__class__.__name__.lower()
    if "torchao" in optimizer_module or optimizer_name.startswith("adamw4bit") or optimizer_name.startswith("adamw8bit"):
        required_state_keys = {"exp_avg", "exp_avg_sq"}
    elif "bitsandbytes" in optimizer_module:
        required_state_keys = {"state1", "state2"}
    elif optimizer_module.startswith("torch.optim"):
        required_state_keys = {"exp_avg", "exp_avg_sq"}
    else:
        required_state_keys = set()

    for group_idx, curr_group in enumerate(current_groups):
        curr_params = curr_group.get('params', [])
        if group_idx >= len(state_groups):
            missing_params += len(curr_params)
            if param_name_map:
                for curr_param_id in curr_params:
                    if len(missing_examples) >= max_examples:
                        break
                    missing_examples.append(param_name_map.get(curr_param_id, f"group{group_idx}"))
            continue

        ckpt_params = state_groups[group_idx].get('params', [])
        if len(ckpt_params) > len(curr_params):
            extra_ckpt_params += len(ckpt_params) - len(curr_params)

        for param_idx, curr_param_id in enumerate(curr_params):
            if param_idx >= len(ckpt_params):
                missing_params += 1
                if param_name_map and len(missing_examples) < max_examples:
                    missing_examples.append(param_name_map.get(curr_param_id, f"group{group_idx}[{param_idx}]"))
                continue

            ckpt_param_id = ckpt_params[param_idx]
            ckpt_state = state.get(ckpt_param_id)
            if not ckpt_state:
                missing_params += 1
                if param_name_map and len(missing_examples) < max_examples:
                    missing_examples.append(param_name_map.get(curr_param_id, f"group{group_idx}[{param_idx}]"))
                continue

            if required_state_keys and not required_state_keys.issubset(ckpt_state.keys()):
                missing_params += 1
                if param_name_map and len(missing_examples) < max_examples:
                    name = param_name_map.get(curr_param_id, f"group{group_idx}[{param_idx}]")
                    missing = sorted(required_state_keys - set(ckpt_state.keys()))
                    missing_examples.append(f"{name} (missing keys: {missing})")
                continue

            param = param_id_to_param.get(curr_param_id)
            if param is None:
                missing_params += 1
                continue

            mismatch = False
            mismatch_info = None
            param_shape_keys = {
                'exp_avg',
                'exp_avg_sq',
                'max_exp_avg_sq',
                'momentum_buffer',
                'state1',
                'state2',
            }
            for key, value in ckpt_state.items():
                if not torch.is_tensor(value):
                    continue
                if key not in param_shape_keys:
                    continue
                if value.shape != param.shape:
                    mismatch = True
                    mismatch_info = (key, tuple(value.shape), tuple(param.shape))
                    break
            if mismatch:
                shape_mismatch += 1
                if len(mismatch_examples) < max_examples:
                    name = None
                    if param_name_map:
                        name = param_name_map.get(curr_param_id)
                    label = name or f"group{group_idx}[{param_idx}]"
                    if mismatch_info is not None:
                        key, ckpt_shape, param_shape = mismatch_info
                        label = f"{label} (key={key}, ckpt={ckpt_shape}, param={param_shape})"
                    mismatch_examples.append(label)
                continue

            loaded_state['state'][curr_param_id] = ckpt_state
            loaded_params += 1

    try:
        optimizer.load_state_dict(loaded_state)
    except Exception as exc:
        logging.warning("Failed to load optimizer state; starting fresh optimizer. Error: %s", exc)
        return False

    logging.info(
        "Optimizer resume: loaded %d/%d params; missing %d; shape mismatch %d; extra ckpt groups %d; extra ckpt params %d.",
        loaded_params,
        total_params,
        missing_params,
        shape_mismatch,
        extra_ckpt_groups,
        extra_ckpt_params,
    )
    if missing_examples:
        logging.info("Optimizer resume missing state sample: %s", missing_examples)
    if mismatch_examples:
        logging.info("Optimizer resume shape-mismatch sample: %s", mismatch_examples)

    return loaded_params > 0


def load_optimizer_state_from_resume(optimizer, optimizer_state, net, logging):
    """Load optimizer state and map param ids to names for better logging."""
    param_to_name = {param: name for name, param in net.named_parameters()}
    optimizer_state_current = optimizer.state_dict()
    param_ids_flat = [
        param_id
        for group in optimizer_state_current.get('param_groups', [])
        for param_id in group.get('params', [])
    ]
    params_flat = [
        param
        for group in optimizer.param_groups
        for param in group.get('params', [])
    ]
    param_name_map = {}
    for param_id, param in zip(param_ids_flat, params_flat):
        name = param_to_name.get(param)
        if name:
            param_name_map[param_id] = name
    return load_optimizer_state(
        optimizer,
        optimizer_state,
        logging,
        param_name_map=param_name_map,
    )


def load_resume_checkpoint(net: torch.nn.Module, configs, logging, tokenizer=None):
    """
    Optionally load model (and optimizer) state from a checkpoint based on config.

    Args:
        net: The model instance to populate.
        configs: Configuration object that may include resume settings.
        logging: Logger for informational output.

    Returns:
        dict or None: Resume metadata with epoch/global_step and optimizer/scheduler
        state dicts, or None if resume is disabled.
    """
    resume_cfg = getattr(configs, 'resume', None)
    if not (resume_cfg and getattr(resume_cfg, 'enable', False)):
        return None

    resume_path = resume_cfg.resume_path
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found at {resume_path}")

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    model_state = _clean_state_dict_keys(model_state)

    if getattr(resume_cfg, 'discard_decoder_weights', False):
        drop_prefixes = ('vqvae.decoder',)
        dropped_entries = {
            key: value
            for key, value in model_state.items()
            if any(key.startswith(prefix) for prefix in drop_prefixes)
        }
        filtered_state = {
            key: value
            for key, value in model_state.items()
            if key not in dropped_entries
        }
        dropped_param_count = sum(
            value.numel() if hasattr(value, 'numel') else 0
            for value in dropped_entries.values()
        )
        model_state = filtered_state
        logging.info(
            "Discarded %s decoder parameters (%s tensors) when resuming from checkpoint",
            f"{dropped_param_count:,}",
            len(dropped_entries),
        )

    current_state = net.state_dict()
    if tokenizer is not None:
        vocab_path = _find_tokenizer_vocab_path(resume_path, resume_cfg)
        if vocab_path:
            old_token_to_id = _load_tokenizer_vocab(vocab_path)
            new_token_to_id = getattr(tokenizer, 'token_to_id', None)
            if old_token_to_id and new_token_to_id:
                logging.info('Using tokenizer vocab from %s for embedding remap.', vocab_path)
                model_state = _remap_embedding_weights(
                    model_state,
                    current_state,
                    old_token_to_id,
                    new_token_to_id,
                    logging,
                )
                model_state = _drop_output_projection_weights(model_state, logging)
            else:
                logging.info('Tokenizer vocab file missing token mappings at %s.', vocab_path)
        else:
            logging.info('No tokenizer_vocab.yaml found for resume; loading weights as-is.')

    filtered_state = {}
    skipped_shape = []
    for key, tensor in model_state.items():
        if key not in current_state:
            continue
        if current_state[key].shape != tensor.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = tensor

    if skipped_shape:
        sample = skipped_shape[:5]
        logging.info(
            'Skipped %d keys due to shape mismatch (sample: %s).',
            len(skipped_shape),
            sample,
        )

    missing, unexpected = net.load_state_dict(filtered_state, strict=False)
    skip_prefixes = ('vqvae.decoder',)
    filtered_missing = [key for key in missing if not key.startswith(skip_prefixes)]
    filtered_unexpected = [key for key in unexpected if not key.startswith(skip_prefixes)]

    if filtered_missing:
        logging.info(
            "Missing keys during resume (%d): %s",
            len(filtered_missing),
            filtered_missing,
        )
    if filtered_unexpected:
        logging.info(
            "Unexpected keys during resume (%d): %s",
            len(filtered_unexpected),
            filtered_unexpected,
        )

    logging.info(f"Resumed model weights from {resume_path}")
    if getattr(resume_cfg, 'restart_optimizer', True):
        optimizer_state = None
        scheduler_state = None
    else:
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if optimizer_state is None:
            logging.info("Checkpoint missing optimizer state; optimizer will start fresh.")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
    }


def _compile_target_enabled(targets, name: str) -> bool:
    if targets is None:
        return True
    if isinstance(targets, dict):
        return bool(targets.get(name, False))
    return bool(getattr(targets, name, False))


def _compile_selected_submodules(
    model: torch.nn.Module,
    compile_kwargs: dict,
    targets=None,
    logging=None,
    compile_mode=None,
):
    net = model
    net._compiled_submodules = False

    compiled_modules = []
    if (
        _compile_target_enabled(targets, "autoregressive_transformer")
        and getattr(net, "transformer", None) is not None
    ):
        transformer = net.transformer
        inner_net = getattr(transformer, "net", None)
        if inner_net is not None:
            transformer.net = torch.compile(inner_net, **compile_kwargs)
        else:
            net.transformer = torch.compile(transformer, **compile_kwargs)
        compiled_modules.append("autoregressive_transformer")

    context_model = getattr(net, "protein_encoder_context_model", None)
    if _compile_target_enabled(targets, "protein_encoder_context_model") and context_model is not None:
        net.protein_encoder_context_model = torch.compile(context_model, **compile_kwargs)
        compiled_modules.append("protein_encoder_context_model")

    context_project = getattr(net, "protein_encoder_context_project", None)
    if (
        _compile_target_enabled(targets, "protein_encoder_context_project")
        and context_project is not None
        and not isinstance(context_project, torch.nn.Identity)
    ):
        net.protein_encoder_context_project = torch.compile(context_project, **compile_kwargs)
        compiled_modules.append("protein_encoder_context_project")

    if compiled_modules:
        # Accelerate FSDP2 checks compiled regions and expects `_orig_mod` when
        # child modules are compiled.
        net.__dict__.setdefault("_orig_mod", net)
        net._compiled_submodules = True
        if logging is not None:
            logging.info("torch.compile mode: %s", compile_mode or "default")
            logging.info("torch.compile submodules: %s", ", ".join(compiled_modules))
    elif logging is not None:
        logging.warning(
            "compile_model=true but no eligible submodules were found; running in eager mode."
        )

    return net


def compile_model(model: torch.nn.Module, configs=None, logging=None, *, mode=None, targets=None):
    """
    Compile model according to call context.

    Training path:
        ``compile_model(model, configs, logging, inference=False)``
        Compiles performance-critical submodules (never wrapped full model).
    """
    # Backward-compatible path for inference/evaluation scripts. If targets are
    # omitted, preserve the old full-module compile behavior.
    if configs is None:
        compile_kwargs = {}
        if mode is not None:
            compile_kwargs["mode"] = mode
        if targets is not None:
            return _compile_selected_submodules(
                model,
                compile_kwargs,
                targets=targets,
                logging=logging,
                compile_mode=mode,
            )
        return torch.compile(model, **compile_kwargs)

    net = model

    if not bool(getattr(configs.model, "compile_model", False)):
        if logging is not None:
            logging.info("torch.compile disabled by config (model.compile_model=false)")
        return net

    compile_mode = mode if mode is not None else getattr(configs.model, "compile_mode", None)
    if compile_mode is None:
        compile_mode = "max-autotune"

    compile_kwargs = {}
    if compile_mode:
        compile_kwargs["mode"] = compile_mode

    return _compile_selected_submodules(
        net,
        compile_kwargs,
        targets=getattr(configs.model, "compile_targets", None),
        logging=logging,
        compile_mode=compile_mode,
    )
