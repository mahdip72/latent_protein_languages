import os
from collections import OrderedDict
import torch
from typing import Optional, Tuple
from accelerate import Accelerator


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
        **kwargs: Additional checkpoint data. Expected key:
            - global_step (int): Total number of training iterations completed.

    Returns:
        None
    """

    with torch.no_grad():
        # Build checkpoint dictionary using a robust path across accelerate versions
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

    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': kwargs['global_step'],
    }, model_path)


def load_resume_checkpoint(net: torch.nn.Module, configs, logging):
    """
    Optionally load model (and optimizer) state from a checkpoint based on config.

    Args:
        net: The model instance to populate.
        configs: Configuration object that may include resume settings.
        logging: Logger for informational output.

    Returns:
        None
    """
    resume_cfg = getattr(configs, 'resume', None)
    if not (resume_cfg and getattr(resume_cfg, 'enable', False)):
        return None, None, None

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

    missing, unexpected = net.load_state_dict(model_state, strict=False)
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

    if not getattr(resume_cfg, 'restart_optimizer', True):
        # raise an error it is not implemented
        ValueError("Resuming optimizer state is not implemented in this version.")


    logging.info(f"Resumed model weights from {resume_path}")


def compile_model(net: torch.nn.Module, *, mode: None | str) -> torch.nn.Module:
    """Compile all model components except the vector-quantizer.

    Mimics the reference project by compiling every submodule with :func:`torch.compile`
    while keeping the vector quantizer in eager mode (it relies on custom autograd).

    Args:
        net: The ``SuperModel`` instance to compile.
        mode: Optional compilation mode for :func:`torch.compile`. e.g. "reduce-overhead", "max-autotune", etc.

    Returns:
        The input model with eligible submodules replaced by compiled versions.
    """

    def _compile_child(module: torch.nn.Module, name: str, child: torch.nn.Module) -> None:
        if child is None or name == 'vector_quantizer':
            return

        if isinstance(child, torch.nn.ModuleList):
            compiled = torch.nn.ModuleList([
                torch.compile(submodule, mode=mode)
                if isinstance(submodule, torch.nn.Module) else submodule
                for submodule in child
            ])
            setattr(module, name, compiled)
            return

        if isinstance(child, torch.nn.ModuleDict):
            compiled = torch.nn.ModuleDict({
                key: torch.compile(submodule, mode=mode)
                if isinstance(submodule, torch.nn.Module) else submodule
                for key, submodule in child.items()
            })
            setattr(module, name, compiled)
            return

        if isinstance(child, torch.nn.Module):
            setattr(module, name, torch.compile(child, mode=mode))

    if hasattr(net, 'protein_encoder') and isinstance(net.protein_encoder, torch.nn.Module):
        net.protein_encoder = torch.compile(net.protein_encoder, mode=mode)

    vq_module = getattr(net, 'vqvae', None)
    if isinstance(vq_module, torch.nn.Module):
        compiled_children = {name for name, _ in vq_module.named_children()}
        for name, child in list(vq_module.named_children()):
            _compile_child(vq_module, name, child)

        extra_attrs = (
            'encoder_tail', 'encoder_head', 'decoder_tail', 'decoder_head',
            'ntp_projector_head', 'ntp_blocks', 'tik_tok_padding_classifier'
        )
        for attr in extra_attrs:
            if attr in compiled_children:
                continue
            child = getattr(vq_module, attr, None)
            if isinstance(child, torch.nn.Module):
                _compile_child(vq_module, attr, child)

    return net
