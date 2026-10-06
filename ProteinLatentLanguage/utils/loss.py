import torch
import torch.nn.functional as F
from typing import Optional
import torch._functorch.config as functorch_config


def calculate_ntp_loss(ntp_logits: Optional[torch.Tensor], indices: Optional[torch.Tensor],
                       valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Vectorized next-token prediction loss using VQ code indices as labels.

    Steps:
    - Apply valid mask to indices; set invalid positions to ignore_index (-100)
    - Shift labels by one (labels[t] = indices[t+1]) and pad last with ignore_index
    - Compute cross-entropy with ignore_index=-100
    - Return per-sample mean loss over non-ignored positions (zeros if none)
    """
    if ntp_logits is None or indices is None:
        if ntp_logits is not None:
            device = ntp_logits.device
        elif indices is not None:
            device = indices.device
        elif valid_mask is not None:
            device = valid_mask.device
        else:
            device = torch.device('cpu')
        batch_size = valid_mask.size(0) if valid_mask is not None else 1
        return torch.zeros(batch_size, device=device)

    ignore_index = -100
    device = ntp_logits.device

    # Ensure dtypes
    indices = indices.to(dtype=torch.long, device=device)
    if valid_mask is None:
        valid_mask = torch.ones_like(indices, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=device)

    B, L, K = ntp_logits.shape

    # Mask invalid positions to ignore_index
    labels_masked = indices.masked_fill(~valid_mask, ignore_index)

    # Shift left by one and pad last as ignore
    pad_col = torch.full((B, 1), ignore_index, dtype=torch.long, device=device)
    labels = torch.cat([labels_masked[:, 1:], pad_col], dim=1)  # (B, L)

    # Flatten for CE
    logits_flat = ntp_logits.reshape(B * L, K)
    labels_flat = labels.reshape(B * L)

    # Per-position loss (ignored positions contribute 0 with 'none' + manual mask)
    loss_flat = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
    loss = loss_flat.view(B, L)

    # Per-sample mean over non-ignored positions
    valid_pos = (labels != ignore_index)
    denom = valid_pos.sum(dim=1).clamp(min=1)
    per_sample_loss = (loss.sum(dim=1) / denom)

    return per_sample_loss


def calculate_loss(model_output, alpha, configs, adaptive_loss_coeffs=None):
    """
    Calculates the total loss, including VQ-VAE auxiliary loss and reconstruction losses.

    Args:
        model_output (dict): Modular output dictionary from SuperModel.forward(), containing:
            - "decoder_output" (torch.Tensor): The reconstructed output from the decoder (B, L, encoder_dim).
            - "encoder_embeddings" (torch.Tensor): The output from the protein encoder (B, L, encoder_dim).
            - "vq_loss" (torch.Tensor): The VQ loss from the vector quantizer (scalar).
            - "mask" (torch.Tensor): Boolean mask for valid positions (B, L).
            - "indices" (torch.Tensor): VQ codebook indices (B, L).
        alpha (float): The weight for the VQ loss.
        configs: The configuration object.

    Optional Args:
        adaptive_loss_coeffs (dict | None): Adaptive scaling coefficients applied in addition to
            config coefficients. Keys: 'mse', 'cosine_similarity', 'cross_entropy', 'vq'.
            Defaults to 1.0 for any missing key or when None.

    Returns:
        dict: A dictionary containing the total loss and its components:
            - "loss" (torch.Tensor): The total combined loss.
            - "rec_loss" (torch.Tensor): The total reconstruction loss.
            - "vq_loss" (torch.Tensor): The VQ-VAE VQ loss.
            - "mse_loss" (torch.Tensor): The mean squared error loss (NaN if disabled).
            - "cosine_loss" (torch.Tensor): The cosine similarity loss (NaN if disabled).
            - "ce_loss" (torch.Tensor): The cross-entropy loss (NaN if disabled).
            - "indices" (torch.Tensor): The indices of the codebook vectors used.
    """
    device = model_output['decoder_output'].device
    vq_loss = model_output['vq_loss']

    classification_cfg = getattr(configs.train_settings.losses, 'classification', None)
    classification_loss_enabled = bool(getattr(classification_cfg, 'enabled', False)) if classification_cfg is not None else False

    # Unscaled raw component losses
    mse_loss, cosine_loss, ce_loss, ntp_loss, unscaled_classification_loss, scaled_classification_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device)
    )

    # Scaled reconstruction loss accumulator
    scaled_rec_loss = 0.0

    # Resolve adaptive coefficients
    adaptive = adaptive_loss_coeffs or {}
    adaptive_mse = float(adaptive.get('mse', 1.0))
    adaptive_cos = float(adaptive.get('cosine_similarity', 1.0))
    adaptive_ce = float(adaptive.get('cross_entropy', 1.0))
    adaptive_vq = float(adaptive.get('vq', 1.0))
    adaptive_ntp = float(adaptive.get('ntp', 1.0))
    adaptive_tik_tok = float(adaptive.get('tik_tok_padding', 1.0))
    adaptive_classification = float(adaptive.get('classification', 1.0))

    # Reconstruction losses
    if configs.train_settings.losses.mse.enabled:
        mse_loss = F.mse_loss(model_output['decoder_output'], model_output['encoder_embeddings'], reduction='none')
        mse_loss = mse_loss[model_output['mask']]
        mse_loss = mse_loss.mean()
        scaled_rec_loss += (
            configs.train_settings.losses.mse.coefficient
            * adaptive_mse
            * mse_loss
        )

    if configs.train_settings.losses.cosine_similarity.enabled:
        cosine_loss = 1 - F.cosine_similarity(model_output['decoder_output'], model_output['encoder_embeddings'],
                                              dim=-1)
        cosine_loss = cosine_loss[model_output['mask']]
        cosine_loss = cosine_loss.mean()

        scaled_rec_loss += (
            configs.train_settings.losses.cosine_similarity.coefficient
            * adaptive_cos
            * cosine_loss
        )

    if configs.train_settings.losses.cross_entropy.enabled:
        reconstructed_log_probs = F.log_softmax(model_output['decoder_output'], dim=-1)
        original_probs = F.softmax(model_output['encoder_embeddings'], dim=-1)

        ce_loss = - (original_probs * reconstructed_log_probs).sum(dim=-1).mean()
        scaled_rec_loss += (
            configs.train_settings.losses.cross_entropy.coefficient
            * adaptive_ce
            * ce_loss
        )

    if classification_loss_enabled:
        logits = model_output.get('decoder_output', None)
        targets = model_output.get('classification_targets', None)
        if logits is None or targets is None:
            raise ValueError(
                "Classification loss is enabled, but decoder outputs or targets are missing from model output."
            )
        logits = logits.to(device)
        targets = targets.to(device)
        classification_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        unscaled_classification_loss = classification_loss
        scaled_classification_loss = (
            float(classification_cfg.coefficient)
            * adaptive_classification
            * classification_loss
        )

    # Next Token Prediction loss (optional)
    if configs.train_settings.losses.next_token_prediction.enabled:
        ntp_loss_vector = calculate_ntp_loss(
            model_output['ntp_logits'],
            model_output['indices'],
            model_output['ntp_mask']
        )
        ntp_loss = ntp_loss_vector.mean()

    # Combine losses. NTP is weighted by config weight and adaptive coefficient
    ntp_weight = float(getattr(getattr(configs.train_settings.losses, 'next_token_prediction', {}), 'weight', 0.0))

    tik_tok_padding_loss = torch.tensor(0.0, device=device)
    scaled_tik_tok_loss = torch.tensor(0.0, device=device)
    tik_tok_cfg = getattr(getattr(configs.model.vqvae.vector_quantization, 'tik_tok', None), 'classifier_weight', None)
    tik_tok_weight = float(tik_tok_cfg) if tik_tok_cfg is not None else 0.0

    if tik_tok_weight > 0.0:
        tik_tok_logits = model_output.get('tik_tok_padding_logits', None)
        tik_tok_targets = model_output.get('tik_tok_padding_targets', None)
        if tik_tok_logits is not None and tik_tok_targets is not None:
            tik_tok_targets = tik_tok_targets.to(dtype=torch.long, device=device)
            tik_tok_loss_vector = F.cross_entropy(
                tik_tok_logits,
                tik_tok_targets,
                reduction='none'
            )
            tik_tok_padding_loss = tik_tok_loss_vector.mean()
            scaled_tik_tok_loss = tik_tok_weight * adaptive_tik_tok * tik_tok_padding_loss

    # Unscaled aggregates
    unscaled_rec_loss = torch.tensor(0.0, device=device)
    if configs.train_settings.losses.mse.enabled:
        unscaled_rec_loss = unscaled_rec_loss + mse_loss
    if configs.train_settings.losses.cosine_similarity.enabled:
        unscaled_rec_loss = unscaled_rec_loss + cosine_loss
    if configs.train_settings.losses.cross_entropy.enabled:
        unscaled_rec_loss = unscaled_rec_loss + ce_loss

    unscaled_total_loss = unscaled_rec_loss + vq_loss + ntp_loss + tik_tok_padding_loss + unscaled_classification_loss

    # Scaled components
    scaled_mse_loss = torch.tensor(0.0, device=device)
    scaled_cosine_loss = torch.tensor(0.0, device=device)
    scaled_ce_loss = torch.tensor(0.0, device=device)
    if configs.train_settings.losses.mse.enabled:
        scaled_mse_loss = (
            configs.train_settings.losses.mse.coefficient
            * adaptive_mse
            * mse_loss
        )
    if configs.train_settings.losses.cosine_similarity.enabled:
        scaled_cosine_loss = (
            configs.train_settings.losses.cosine_similarity.coefficient
            * adaptive_cos
            * cosine_loss
        )
    if configs.train_settings.losses.cross_entropy.enabled:
        scaled_ce_loss = (
            configs.train_settings.losses.cross_entropy.coefficient
            * adaptive_ce
            * ce_loss
        )
    # Recompute scaled_rec_loss precisely as sum of components (for logging clarity)
    scaled_rec_loss = scaled_mse_loss + scaled_cosine_loss + scaled_ce_loss
    scaled_vq_loss = (alpha * adaptive_vq) * vq_loss
    scaled_ntp_loss = (ntp_weight * adaptive_ntp) * ntp_loss
    scaled_total_loss = scaled_rec_loss + scaled_vq_loss + scaled_ntp_loss + scaled_tik_tok_loss + scaled_classification_loss

    return {
        # Scaled for training (backward)
        "total_loss": scaled_total_loss,
        "rec_loss": scaled_rec_loss,
        "vq_loss": scaled_vq_loss,  # raw vq loss kept for compatibility
        "mse_loss": scaled_mse_loss,  # raw component
        "cosine_loss": scaled_cosine_loss,  # raw component
        "ce_loss": scaled_ce_loss,  # raw component
        "ntp_loss": scaled_ntp_loss,  # raw component
        "tik_tok_padding_loss": scaled_tik_tok_loss,
        "classification_loss": scaled_classification_loss,
        "indices": model_output["indices"],

        # Explicit unscaled aggregates
        "unscaled_total_loss": unscaled_total_loss,
        "unscaled_rec_loss": unscaled_rec_loss,
        "unscaled_vq_loss": vq_loss,
        "unscaled_ntp_loss": ntp_loss,
        "unscaled_tik_tok_padding_loss": tik_tok_padding_loss,
        "unscaled_mse_loss": mse_loss,
        "unscaled_cosine_loss": cosine_loss,
        "unscaled_ce_loss": ce_loss,
        "unscaled_classification_loss": unscaled_classification_loss,
    }


def compute_grad_norm(loss, parameters, norm_type=2):
    """
    Compute the gradient norm for a given loss and model parameters without altering existing gradients.

    Args:
        loss (torch.Tensor): The loss tensor.
        parameters (iterable): Iterable of model parameters.
        norm_type (float): The type of norm (default 2 for L2 norm).

    Returns:
        torch.Tensor: The gradient norm.
    """
    trainable_params = [p for p in parameters if p.requires_grad]
    with functorch_config.patch(donated_buffer=False):
        grads = torch.autograd.grad(
            loss,
            trainable_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
    grads = [g for g in grads if g is not None]
    if not grads:
        device = getattr(loss, "device", None)
        if device is None or device.type == "cpu":
            if trainable_params:
                device = trainable_params[0].device
            elif torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        return torch.tensor(0.0, device=device)
    norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return norm


def adjust_coeff_by_grad(coeff, grad_norm, decrease_factor=0.98, increase_factor=1.02, upper_thresh=2.0, lower_thresh=0.05):
    """
    Adjust a coefficient based on gradient norm magnitude.

    Args:
        coeff (float): Current coefficient value.
        grad_norm (float): Gradient norm of the corresponding loss component.
        decrease_factor (float): Factor to multiply coeff by if grad_norm > upper_thresh.
        increase_factor (float): Factor to multiply coeff by if grad_norm < lower_thresh.
        upper_thresh (float): Threshold above which to decrease coefficient.
        lower_thresh (float): Threshold below which to increase coefficient.

    Returns:
        float: Adjusted coefficient.
    """
    if grad_norm > upper_thresh:
        return coeff * decrease_factor
    elif grad_norm < lower_thresh:
        return coeff * increase_factor
    else:
        return coeff


def adjust_adaptive_coefficients(adaptive_loss_coeffs, global_grad_norms, configs):
    """
    Adjust adaptive loss coefficients based on global gradient norms and configuration.

    - Respects per-loss `adaptive_coefficient` toggles for: MSE, cosine similarity, cross-entropy, and VQ.
    - Uses global gradient norms aggregated across ranks to decide whether to increase/decrease coefficients.
    - Clamps coefficients to `[train_settings.adaptive_coefficient_bounds.min, train_settings.adaptive_coefficient_bounds.max]`
      when those bounds are present in the config.

    Args:
        adaptive_loss_coeffs (dict): Current adaptive coefficients keyed by
            `mse`, `cosine_similarity`, `cross_entropy`, `vq`.
        global_grad_norms (dict): Global gradient norms keyed by
            `mse`, `cosine`, `ce`, `vq`.
        configs: Configuration object containing per-loss toggles and optional bounds.

    Returns:
        dict: Updated adaptive coefficients.
    """
    # Adjust each coefficient based on its global grad norm
    if 'mse' in global_grad_norms and getattr(
        configs.train_settings.losses.mse, 'adaptive_coefficient', False
    ):
        adaptive_loss_coeffs['mse'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['mse'], global_grad_norms['mse']
        )

    if 'cosine' in global_grad_norms and getattr(
        configs.train_settings.losses.cosine_similarity, 'adaptive_coefficient', False
    ):
        adaptive_loss_coeffs['cosine_similarity'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['cosine_similarity'], global_grad_norms['cosine']
        )

    if 'ce' in global_grad_norms and getattr(
        configs.train_settings.losses.cross_entropy, 'adaptive_coefficient', False
    ):
        adaptive_loss_coeffs['cross_entropy'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['cross_entropy'], global_grad_norms['ce']
        )


    if 'classification' in global_grad_norms and getattr(
        getattr(configs.train_settings.losses, 'classification', {}), 'adaptive_coefficient', False
    ):
        adaptive_loss_coeffs['classification'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs.get('classification', 1.0), global_grad_norms['classification']
        )

    vq_cfg = getattr(configs.model.vqvae, 'vector_quantization', None)
    if (
        'vq' in global_grad_norms
        and getattr(vq_cfg, 'adaptive_coefficient', False)
        and not getattr(vq_cfg, 'freeze_parameters', False)
    ):
        adaptive_loss_coeffs['vq'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['vq'], global_grad_norms['vq']
        )

    if 'ntp' in global_grad_norms and getattr(
        getattr(configs.train_settings.losses, 'next_token_prediction', {}), 'adaptive_coefficient', False
    ):
        adaptive_loss_coeffs['ntp'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs.get('ntp', 1.0), global_grad_norms['ntp']
        )

    tik_tok_cfg = getattr(configs.model.vqvae.vector_quantization, 'tik_tok', None)
    if (
        'tik_tok_padding' in global_grad_norms
        and tik_tok_cfg is not None
        and getattr(tik_tok_cfg, 'adaptive_coefficient', False)
    ):
        adaptive_loss_coeffs['tik_tok_padding'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs.get('tik_tok_padding', 1.0), global_grad_norms['tik_tok_padding']
        )


    min_coeff = float(configs.train_settings.adaptive_coefficient_bounds.min)
    max_coeff = float(configs.train_settings.adaptive_coefficient_bounds.max)


    for key in ['mse', 'cosine_similarity', 'cross_entropy', 'vq', 'ntp', 'tik_tok_padding', 'classification']:
        if key in adaptive_loss_coeffs:
            adaptive_loss_coeffs[key] = float(
                max(min_coeff, min(max_coeff, adaptive_loss_coeffs[key]))
            )

    return adaptive_loss_coeffs


def log_gradient_norms_and_coeffs(writer, global_grad_norms, adaptive_loss_coeffs, global_step):
    """
    Log gradient norms and adaptive coefficients to TensorBoard.

    Note: This function always logs the current coefficient values, regardless of whether
    adaptive_loss_coefficient is enabled or not. This allows monitoring coefficient
    evolution when adaptive mode is on, and confirms they remain constant when off.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        global_grad_norms (dict): Global gradient norms for each loss component.
        adaptive_loss_coeffs (dict): Current adaptive coefficients.
        global_step (int): Current global training step.
    """
    # Log gradient norms
    for key, norm in global_grad_norms.items():
        if key == 'mse':
            writer.add_scalar('gradient norm/mse', norm, global_step)
        elif key == 'cosine':
            writer.add_scalar('gradient norm/cosine', norm, global_step)
        elif key == 'ce':
            writer.add_scalar('gradient norm/ce', norm, global_step)
        elif key == 'classification':
            writer.add_scalar('gradient norm/classification', norm, global_step)
        elif key == 'vq':
            writer.add_scalar('gradient norm/vq', norm, global_step)
        elif key == 'ntp':
            writer.add_scalar('gradient norm/ntp', norm, global_step)
        elif key == 'tik_tok_padding':
            writer.add_scalar('gradient norm/tik_tok_padding', norm, global_step)
        elif key == 'total_unscaled':
            writer.add_scalar('gradient norm/total_unscaled', norm, global_step)

    # Log adaptive coefficients
    for coeff_name, coeff_val in adaptive_loss_coeffs.items():
        writer.add_scalar(f'adaptive_coeff/{coeff_name}', coeff_val, global_step)


def log_per_loss_components(writer, loss_dict, global_step):
    """
    Log individual loss components to TensorBoard.

    Two categories:
    - step_loss/*: scaled components using legacy keys ('loss', 'rec_loss', etc.)
    - unscaled_step_loss/*: raw components using 'unscaled_*' keys

    Args:
        writer (SummaryWriter): TensorBoard writer.
        loss_dict (dict): Dictionary with legacy keys and unscaled_* keys.
        global_step (int): Current global training step.
    """
    legacy_map = {
        'total': 'total_loss',
        'rec': 'rec_loss',
        'vq': 'vq_loss',
        'mse': 'mse_loss',
        'cosine': 'cosine_loss',
        'ce': 'ce_loss',
        'ntp': 'ntp_loss',
        'tik_tok_padding': 'tik_tok_padding_loss',
        'classification': 'classification_loss',
    }

    unscaled_map = {
        'total': 'unscaled_total_loss',
        'rec': 'unscaled_rec_loss',
        'vq': 'unscaled_vq_loss',
        'mse': 'unscaled_mse_loss',
        'cosine': 'unscaled_cosine_loss',
        'ce': 'unscaled_ce_loss',
        'ntp': 'unscaled_ntp_loss',
        'tik_tok_padding': 'unscaled_tik_tok_padding_loss',
        'classification': 'unscaled_classification_loss',
    }

    for name, legacy_key in legacy_map.items():
        # Scaled (legacy) step loss
        scaled_val = loss_dict.get(legacy_key, None)
        if scaled_val is not None:
            if isinstance(scaled_val, torch.Tensor):
                scaled_val = scaled_val.detach().item()
            writer.add_scalar(f'step_loss/{name}', float(scaled_val), global_step)

        # Unscaled step loss
        unscaled_key = unscaled_map[name]
        unscaled_val = loss_dict.get(unscaled_key, None)
        if unscaled_val is not None:
            if isinstance(unscaled_val, torch.Tensor):
                unscaled_val = unscaled_val.detach().item()
            writer.add_scalar(f'unscaled_step_loss/{name}', float(unscaled_val), global_step)


def aggregate_grad_norms(local_grad_norms, accelerator):
    """
    Aggregate local gradient norms across all ranks to get global signals.

    Args:
        local_grad_norms (dict): Local gradient norms computed on this rank.
        accelerator: Hugging Face Accelerator for gathering.

    Returns:
        dict: Global gradient norms (averaged across ranks).
    """
    global_grad_norms = {}
    for key, local_norm in local_grad_norms.items():
        gathered_norms = accelerator.gather_for_metrics(local_norm)
        global_grad_norms[key] = gathered_norms.mean().item()
    return global_grad_norms


def broadcast_coefficients(adaptive_loss_coeffs, accelerator):
    """
    Broadcast updated coefficients from main process to all ranks.

    Args:
        adaptive_loss_coeffs (dict): Coefficients to broadcast (updated on main process).
        accelerator: Hugging Face Accelerator.

    Returns:
        dict: Updated coefficients (same on all ranks after broadcast).
    """
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if accelerator.is_main_process:
            obj_list = [adaptive_loss_coeffs]
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0)
        adaptive_loss_coeffs = obj_list[0]
    return adaptive_loss_coeffs


def log_per_loss_grad_norms(loss_batch, net, configs, writer, accelerator, global_step, adaptive_loss_coeffs):
    """
    Compute and log per-loss gradient norms (MSE, cosine, CE, VQ) and adapt coefficients.

    Behavior:
    - Runs only on steps where: gradients are synchronized, `log_separate_grad_norms` is True, and
      `global_step % gradient_norm_logging_freq == 0`.
    - Computes local grad norms for enabled losses, aggregates them across ranks, then:
      - Adjusts adaptive coefficients on the main process only after warmup, when
        `adaptive_loss_coefficient` is True and at least one loss produced a grad norm.
      - Logs gradient norms, current adaptive coefficients, and individual loss components.
      - Broadcasts updated coefficients to all ranks after warmup when adaptation is enabled and we had norms.

    Args:
        loss_batch (dict): Output of `calculate_loss` with keys like `loss`, `rec_loss`, `vq_loss`,
            `mse_loss`, `cosine_loss`, `ce_loss`, and `indices` (computed from model_output).
        net (torch.nn.Module): The model whose parameters are used to compute grad norms.
        configs: Configuration with training settings and toggles.
        writer (SummaryWriter): TensorBoard writer (used on main process for logging).
        accelerator: Hugging Face Accelerator for sync and distributed ops.
        global_step (int): Current global training step.
        adaptive_loss_coeffs (dict): Current adaptive coefficients.

    Returns:
        dict: Possibly updated `adaptive_loss_coeffs` after adjustment/broadcast, or unchanged if guard conditions fail.
    """
    # Early return if not on gradient sync boundary or not logging step
    if not (
        accelerator.sync_gradients
        and configs.train_settings.log_separate_grad_norms
        and global_step % configs.train_settings.gradient_norm_logging_freq == 0
    ):
        return adaptive_loss_coeffs

    # Compute local grad norms on all ranks
    local_grad_norms = {}
    classification_cfg = getattr(configs.train_settings.losses, 'classification', None)

    if configs.train_settings.losses.mse.enabled:
        local_grad_norms['mse'] = compute_grad_norm(loss_batch['mse_loss'], net.parameters())

    if configs.train_settings.losses.cosine_similarity.enabled:
        local_grad_norms['cosine'] = compute_grad_norm(loss_batch['cosine_loss'], net.parameters())

    if configs.train_settings.losses.cross_entropy.enabled:
        local_grad_norms['ce'] = compute_grad_norm(loss_batch['ce_loss'], net.parameters())

    if classification_cfg is not None and getattr(classification_cfg, 'enabled', False):
        local_grad_norms['classification'] = compute_grad_norm(loss_batch['classification_loss'], net.parameters())

    if configs.model.vqvae.vector_quantization.enabled:
        local_grad_norms['vq'] = compute_grad_norm(loss_batch['vq_loss'], net.parameters())

    if configs.train_settings.losses.next_token_prediction.enabled:
        local_grad_norms['ntp'] = compute_grad_norm(loss_batch.get('ntp_loss', torch.tensor(0.0, device=next(net.parameters()).device if any(p.requires_grad for p in net.parameters()) else 'cpu')), net.parameters())

    if configs.model.vqvae.vector_quantization.tik_tok.enabled:
        tik_tok_loss = loss_batch.get('tik_tok_padding_loss', None)
        if isinstance(tik_tok_loss, torch.Tensor) and tik_tok_loss.requires_grad:
            local_grad_norms['tik_tok_padding'] = compute_grad_norm(tik_tok_loss, net.parameters())

    # Unscaled total gradient norm from the combined loss (before backward)
    local_grad_norms['total_unscaled'] = compute_grad_norm(loss_batch['total_loss'], net.parameters())

    # Aggregate grad norms across all ranks to get global signal
    global_grad_norms = aggregate_grad_norms(local_grad_norms, accelerator)

    # Log gradient norms, coefficients, and individual loss components (main process only)
    if accelerator.is_main_process:
        if configs.tensorboard_log:
            log_per_loss_components(writer, loss_batch, global_step)

    # Adjust coefficients (only after warmup and if any adaptive coefficients are enabled)
    if (
        accelerator.is_main_process
        and global_step > configs.optimizer.decay.warmup
        and configs.train_settings.adaptive_loss_coefficient
        and len(local_grad_norms) > 0
    ):
        adaptive_loss_coeffs = adjust_adaptive_coefficients(
            adaptive_loss_coeffs, global_grad_norms, configs
        )

    # Log gradient norms, coefficients, and individual loss components (main process only)
    if accelerator.is_main_process:
        if configs.tensorboard_log:
            log_gradient_norms_and_coeffs(writer, global_grad_norms, adaptive_loss_coeffs, global_step)

    # Broadcast updated coefficients to all ranks (only after warmup and if any adaptive coefficients are enabled)
    if (
        configs.train_settings.adaptive_loss_coefficient
        and global_step > configs.optimizer.decay.warmup
        and len(local_grad_norms) > 0
    ):
        adaptive_loss_coeffs = broadcast_coefficients(adaptive_loss_coeffs, accelerator)

    return adaptive_loss_coeffs
