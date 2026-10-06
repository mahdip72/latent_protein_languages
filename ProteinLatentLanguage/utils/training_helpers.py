from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

import torch
from accelerate import Accelerator
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity

from utils.metrics import NMSE, ReconstructionCrossEntropyDelta, update_ntp_perplexity


@dataclass
class PhaseMetrics:
    """Container for all metric trackers required during a train/eval phase.

    Attributes:
        nmse: Normalised MSE tracker (disabled when the classifier head is active).
        recon_ce: Reconstruction cross-entropy delta tracker.
        classification_micro: Micro-averaged classification accuracy (optional).
        classification_per_class: Per-class classification accuracy (optional).
        class_labels: Labels used to index per-class accuracies.
        ntp_perplexity: Perplexity tracker for next-token prediction (optional).
        tik_tok_accuracy: Accuracy tracker for TikTok padding classifier (optional).
    """

    nmse: Optional[NMSE]
    recon_ce: Optional[ReconstructionCrossEntropyDelta]
    classification_micro: Optional[MulticlassAccuracy]
    classification_per_class: Optional[MulticlassAccuracy]
    class_labels: List[str]
    ntp_perplexity: Optional[Perplexity]
    tik_tok_accuracy: Optional[MulticlassAccuracy]


def init_phase_metrics(configs, accelerator: Accelerator, class_candidates: Sequence[str]) -> PhaseMetrics:
    """Instantiate every metric required by the current configuration.

    Args:
        configs: Hydrated config object with model and loss toggles.
        accelerator: ``accelerate.Accelerator`` used for device placement.
        class_candidates: Ordered list of amino-acid labels used for per-class accuracy.

    Returns:
        ``PhaseMetrics`` with all active metric trackers moved onto ``accelerator.device``.
    """
    classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
    classifier_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False

    nmse: Optional[NMSE] = None
    recon_ce: Optional[ReconstructionCrossEntropyDelta] = None
    class_labels: List[str] = []
    classification_micro: Optional[MulticlassAccuracy] = None
    classification_per_class: Optional[MulticlassAccuracy] = None

    if classifier_enabled:
        num_classes = int(getattr(classifier_cfg, 'num_classes', 1))
        class_labels = list(class_candidates[:num_classes])
        classification_micro = MulticlassAccuracy(
            num_classes=num_classes,
            average='micro',
            ignore_index=-100,
        ).to(accelerator.device)
        classification_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average='none',
            ignore_index=-100,
        ).to(accelerator.device)
    else:
        nmse = NMSE(dist_sync_on_step=False).to(accelerator.device)
        recon_ce = ReconstructionCrossEntropyDelta(dist_sync_on_step=False).to(accelerator.device)

    ntp_cfg = getattr(getattr(configs.train_settings, 'losses', None), 'next_token_prediction', None)
    ntp_enabled = bool(getattr(ntp_cfg, 'enabled', False))
    ntp_perplexity: Optional[Perplexity] = None
    if ntp_enabled:
        ntp_perplexity = Perplexity(ignore_index=-100).to(accelerator.device)

    tik_tok_accuracy: Optional[MulticlassAccuracy] = None
    tik_tok_cfg = getattr(configs.model.vqvae.vector_quantization, 'tik_tok', None)
    compression_factor = getattr(tik_tok_cfg, 'compression_factor', 1) if tik_tok_cfg is not None else 1
    if tik_tok_cfg is not None and getattr(tik_tok_cfg, 'enabled', False) and compression_factor > 1:
        tik_tok_accuracy = MulticlassAccuracy(num_classes=int(compression_factor), average='macro').to(
            accelerator.device
        )

    return PhaseMetrics(
        nmse=nmse,
        recon_ce=recon_ce,
        classification_micro=classification_micro,
        classification_per_class=classification_per_class,
        class_labels=class_labels,
        ntp_perplexity=ntp_perplexity,
        tik_tok_accuracy=tik_tok_accuracy,
    )


def update_phase_metrics(metrics: PhaseMetrics, model_output: MutableMapping[str, torch.Tensor],
                         detach: bool = True) -> None:
    """Update all active metrics with the current model outputs.

    Args:
        metrics: ``PhaseMetrics`` returned by :func:`init_phase_metrics`.
        model_output: Output dictionary produced by the model / loss function.
        detach: Whether to detach tensors before updating (``True`` during training).
    """
    if metrics.nmse is not None:
        decoder_output = model_output['decoder_output']
        encoder_embeddings = model_output['encoder_embeddings']
        mask = model_output['mask']
        if detach:
            decoder_output = decoder_output.detach()
            encoder_embeddings = encoder_embeddings.detach()
            mask = mask.detach()
        metrics.nmse(decoder_output, encoder_embeddings, mask=mask)

    if metrics.recon_ce is not None:
        encoder_embeddings = model_output['encoder_embeddings']
        decoder_output = model_output['decoder_output']
        mask = model_output['mask']
        if detach:
            encoder_embeddings = encoder_embeddings.detach()
            decoder_output = decoder_output.detach()
            mask = mask.detach()
        metrics.recon_ce(encoder_embeddings, decoder_output, mask)

    if metrics.classification_micro is not None and metrics.classification_per_class is not None:
        logits = model_output.get('decoder_output', None)
        targets = model_output.get('classification_targets', None)
        if logits is not None and targets is not None:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1).to(dtype=torch.long)
            if detach:
                flat_logits = flat_logits.detach()
                flat_targets = flat_targets.detach()
            metrics.classification_micro.update(flat_logits, flat_targets)
            metrics.classification_per_class.update(flat_logits, flat_targets)

    if metrics.ntp_perplexity is not None and model_output.get('ntp_logits', None) is not None:
        update_ntp_perplexity(
            metrics.ntp_perplexity,
            model_output['ntp_logits'],
            model_output['indices'],
            model_output.get('ntp_mask', model_output['mask']),
            ignore_index=-100,
        )

    if metrics.tik_tok_accuracy is not None:
        tik_tok_logits = model_output.get('tik_tok_padding_logits', None)
        tik_tok_targets = model_output.get('tik_tok_padding_targets', None)
        if tik_tok_logits is not None and tik_tok_targets is not None and tik_tok_targets.numel() > 0:
            if detach:
                tik_tok_logits = tik_tok_logits.detach()
                tik_tok_targets = tik_tok_targets.detach()
            metrics.tik_tok_accuracy.update(tik_tok_logits, tik_tok_targets)


def summarize_phase_metrics(metrics: PhaseMetrics) -> Dict[str, Any]:
    """Compute scalar summaries from metric state.

    Args:
        metrics: ``PhaseMetrics`` whose trackers contain accumulated statistics.

    Returns:
        Dictionary with scalar values for NMSE, reconstruction CE deltas, classification
        accuracy, per-class classification accuracy, NTP perplexity, and TikTok accuracy.
        Missing metrics are filled with ``float('nan')`` and empty mappings.
    """
    summary: Dict[str, Any] = {
        'nmse_median': float('nan'),
        'nmse_p95': float('nan'),
        'nmse_mean': float('nan'),
        'nmse_strict_coverage': float('nan'),
        'nmse_token_count': 0,
        'recon_ppl_ratio': float('nan'),
        'recon_kl': float('nan'),
        'recon_ppl_pct_delta': float('nan'),
        'classification_accuracy': float('nan'),
        'classification_accuracy_per_class': {},
        'ntp_perplexity': float('nan'),
        'tik_tok_padding_accuracy': float('nan'),
    }

    if metrics.nmse is not None:
        nmse_values = metrics.nmse.compute()
        if isinstance(nmse_values, dict):
            def _to_float(value):
                if value is None:
                    return None
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return None
                    return float(value.detach().cpu().item())
                return float(value)

            def _to_int(value):
                if value is None:
                    return None
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return None
                    return int(value.detach().cpu().item())
                return int(value)

            median = _to_float(nmse_values.get("median"))
            p95 = _to_float(nmse_values.get("p95"))
            mean = _to_float(nmse_values.get("mean"))
            coverage = _to_float(nmse_values.get("strict_coverage"))
            count = _to_int(nmse_values.get("count"))

            summary['nmse_median'] = median if median is not None else float('nan')
            summary['nmse_p95'] = p95 if p95 is not None else float('nan')
            summary['nmse_mean'] = mean if mean is not None else float('nan')
            summary['nmse_strict_coverage'] = coverage if coverage is not None else float('nan')
            summary['nmse_token_count'] = count if count is not None else 0

    if metrics.recon_ce is not None:
        recon_ce_values = metrics.recon_ce.compute()
        summary['recon_ppl_ratio'] = recon_ce_values["ppl_ratio"].item()
        summary['recon_kl'] = recon_ce_values["kl"].item()
        ratio = summary['recon_ppl_ratio']
        summary['recon_ppl_pct_delta'] = (ratio - 1.0) * 100.0 if ratio == ratio else float('nan')

    if metrics.classification_micro is not None:
        acc_tensor = metrics.classification_micro.compute()
        summary['classification_accuracy'] = acc_tensor.item() if acc_tensor is not None else float('nan')
        per_class_tensor = metrics.classification_per_class.compute() if metrics.classification_per_class else None
        if per_class_tensor is not None:
            per_class_list = per_class_tensor.detach().cpu().tolist()
            per_class_map = {
                metrics.class_labels[idx]: float(value)
                for idx, value in enumerate(per_class_list)
                if idx < len(metrics.class_labels)
            }
            summary['classification_accuracy_per_class'] = per_class_map

    if metrics.ntp_perplexity is not None:
        try:
            summary['ntp_perplexity'] = metrics.ntp_perplexity.compute().item()
        except (RuntimeError, ValueError):
            summary['ntp_perplexity'] = float('nan')

    if metrics.tik_tok_accuracy is not None:
        acc_tensor = metrics.tik_tok_accuracy.compute()
        summary['tik_tok_padding_accuracy'] = acc_tensor.item() if acc_tensor is not None else float('nan')

    return summary


LOSS_KEYS: Tuple[str, ...] = (
    'total_loss',
    'rec_loss',
    'vq_loss',
    'mse_loss',
    'cosine_loss',
    'ce_loss',
    'ntp_loss',
    'tik_tok_padding_loss',
    'classification_loss',
)

UNSCALED_KEYS: Tuple[str, ...] = tuple(f'unscaled_{name}' for name in LOSS_KEYS)


def init_loss_tracker(device: torch.device, accum_iter: int) -> Dict[str, Any]:
    """Create a container that mirrors the manual loss bookkeeping in the loops.

    Args:
        device: Device used to store per-accumulation tensors before reduction.
        accum_iter: Gradient accumulation factor (micro-batches per optimizer step).

    Returns:
        Dict containing running tensors for the current accumulation window,
        epoch-level totals, the set of unique indices, and a finalized step counter.
    """
    tracker: Dict[str, Any] = {
        'accum_iter': accum_iter,
        'micro': {key: torch.zeros(1, device=device) for key in LOSS_KEYS},
        'micro_unscaled': {key: torch.zeros(1, device=device) for key in UNSCALED_KEYS},
        'epoch': {key: 0.0 for key in LOSS_KEYS},
        'epoch_unscaled': {key: 0.0 for key in UNSCALED_KEYS},
        'counter': 0,
        'unique_indices': set(),  # type: Set[int]
    }
    return tracker


def _safe_get(loss_batch: MutableMapping[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    value = loss_batch.get(key, None)
    if value is None:
        return torch.zeros(1, device=device)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    value = value.detach().to(device)
    if value.dim() == 0:
        return value.unsqueeze(0)
    if value.numel() == 1:
        return value.view(1)
    return value.view(-1).mean().unsqueeze(0)


def accumulate_train_losses(tracker: Dict[str, Any], loss_batch: MutableMapping[str, torch.Tensor],
                            device: torch.device) -> None:
    """Accumulate losses for the current micro-batch before reduction.

    Args:
        tracker: Dict produced by :func:`init_loss_tracker`.
        loss_batch: Output of ``calculate_loss`` containing scaled/unscaled components.
        device: Device used for temporary tensors.
    """
    for key in LOSS_KEYS:
        tracker['micro'][key] += _safe_get(loss_batch, key, device)
    for key in UNSCALED_KEYS:
        tracker['micro_unscaled'][key] += _safe_get(loss_batch, key, device)


def finalize_train_step(tracker: Dict[str, Any], accelerator: Accelerator,
                        indices: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Reduce accumulated losses across processes, update epoch totals and unique codes.

    Args:
        tracker: Dict produced by :func:`init_loss_tracker`.
        accelerator: Distributed accelerator used for reduction.
        indices: Codebook indices tensor gathered to compute activation statistics.

    Returns:
        Tuple ``(scaled, unscaled)`` where each element is a dict of per-step averages
        keyed by their original tensor names (``'total_loss'`` / ``'unscaled_total_loss'`` etc.).
    """
    accum_iter = max(1, tracker['accum_iter'])

    scaled: Dict[str, float] = {}
    for key, tensor in tracker['micro'].items():
        reduced = accelerator.reduce(tensor, reduction='mean').item() / accum_iter
        tracker['epoch'][key] += reduced
        scaled[key] = reduced
        tracker['micro'][key] = torch.zeros_like(tensor)

    unscaled: Dict[str, float] = {}
    for key, tensor in tracker['micro_unscaled'].items():
        reduced = accelerator.reduce(tensor, reduction='mean').item() / accum_iter
        tracker['epoch_unscaled'][key] += reduced
        unscaled[key] = reduced
        tracker['micro_unscaled'][key] = torch.zeros_like(tensor)

    gathered_indices = accelerator.gather_for_metrics(indices.detach())
    tracker['unique_indices'].update(gathered_indices.unique().cpu().tolist())
    tracker['counter'] += 1

    return scaled, unscaled


def accumulate_eval_losses(tracker: Dict[str, Any], loss_batch: MutableMapping[str, torch.Tensor],
                           accelerator: Accelerator, repeat: int, device: torch.device) -> None:
    """Gather and average unscaled losses for evaluation.

    Args:
        tracker: Dict produced by :func:`init_loss_tracker`.
        loss_batch: Output of ``calculate_loss``.
        accelerator: Distributed accelerator used to gather losses.
        repeat: Number of times to repeat gathered tensors (mirrors original averaging).
        device: Device used for creating fallback tensors.
    """
    for loss_key, metric_key in zip(LOSS_KEYS, UNSCALED_KEYS):
        tensor = loss_batch.get(metric_key, None)
        if tensor is None:
            continue
        value = tensor.detach()
        if value.dim() == 0:
            value = value.repeat(repeat)
        gathered = accelerator.gather(value)
        mean_value = gathered.mean().item()
        tracker['epoch_unscaled'][metric_key] += mean_value
        tracker['epoch'][loss_key] += mean_value


def update_unique_indices(tracker: Dict[str, Any], indices: torch.Tensor, accelerator: Accelerator) -> None:
    """Gather codebook indices across ranks and record the unique set."""
    gathered_indices = accelerator.gather_for_metrics(indices.detach())
    tracker['unique_indices'].update(gathered_indices.unique().cpu().tolist())


def compute_activation_ratio(unique_indices: Set[int], codebook_size: int) -> float:
    """Return the fraction of active codebook entries."""
    if codebook_size <= 0:
        return 0.0
    return len(unique_indices) / float(codebook_size)


def summarize_running_losses(tracker: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return current epoch averages for scaled and unscaled losses."""
    denom = max(1, tracker['counter'])
    scaled = {key: tracker['epoch'][key] / denom for key in LOSS_KEYS}
    unscaled = {key: tracker['epoch_unscaled'][key] / denom for key in UNSCALED_KEYS}
    return scaled, unscaled


def compute_epoch_loss_averages(tracker: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute final epoch averages after all steps are processed."""
    denom = max(1, tracker['counter'])
    scaled = {key: tracker['epoch'][key] / denom for key in LOSS_KEYS}
    unscaled = {key: tracker['epoch_unscaled'][key] / denom for key in UNSCALED_KEYS}
    return scaled, unscaled


def apply_loss_toggles(scaled: Dict[str, float], unscaled: Dict[str, float], toggles: Dict[str, bool]) -> None:
    """Replace disabled loss entries with ``NaN`` in scaled/unscaled dicts."""
    for key, enabled in toggles.items():
        if enabled:
            continue
        if key in scaled:
            scaled[key] = float('nan')
        unscaled_key = f'unscaled_{key}'
        if unscaled_key in unscaled:
            unscaled[unscaled_key] = float('nan')


def format_progress_postfix(optimizer, loss_batch: MutableMapping[str, torch.Tensor], global_step: int) -> Dict[str, float]:
    """Create the tqdm postfix dictionary used in both loops."""
    def _to_scalar(value: Optional[torch.Tensor]) -> float:
        if value is None:
            return float('nan')
        try:
            return float(value.detach().item())
        except (AttributeError, ValueError):
            return float(value)

    return {
        'lr': optimizer.param_groups[0]['lr'],
        'total_loss': _to_scalar(loss_batch.get('total_loss')),
        'rec_loss': _to_scalar(loss_batch.get('rec_loss')),
        'global_step': float(global_step),
    }


def describe_train_progress(epoch: int, scaled_losses: Dict[str, float], tik_tok_acc: float) -> str:
    """Build the descriptive string displayed during the training loop."""
    return (
        f"epoch {epoch} "
        f"[loss: {scaled_losses['total_loss']:.3f}, "
        f"rec loss: {scaled_losses['rec_loss']:.3f}, "
        f"vq loss: {scaled_losses['vq_loss']:.3f}, "
        f"mse loss: {scaled_losses['mse_loss']:.3f}, "
        f"cosine loss: {scaled_losses['cosine_loss']:.3f}, "
        f"ce loss: {scaled_losses['ce_loss']:.3f}, "
        f"ntp loss: {scaled_losses['ntp_loss']:.3f}, "
        f"tik tok loss: {scaled_losses['tik_tok_padding_loss']:.3f}, "
        f"tik tok acc: {tik_tok_acc:.2f}]"
    )


def describe_eval_progress(mode: str, name: str, scaled_losses: Dict[str, float], tik_tok_acc: float) -> str:
    """Build the descriptive string displayed during evaluation/validation."""
    return (
        f"{mode} {name} "
        f"[loss: {scaled_losses['total_loss']:.3f}, "
        f"rec loss: {scaled_losses['rec_loss']:.3f}, "
        f"vq loss: {scaled_losses['vq_loss']:.3f}, "
        f"mse loss: {scaled_losses['mse_loss']:.3f}, "
        f"cosine loss: {scaled_losses['cosine_loss']:.3f}, "
        f"ce loss: {scaled_losses['ce_loss']:.3f}, "
        f"ntp loss: {scaled_losses['ntp_loss']:.3f}, "
        f"tik tok loss: {scaled_losses['tik_tok_padding_loss']:.3f}, "
        f"tik tok acc: {tik_tok_acc:.2f}]"
    )


def log_tensorboard_phase(writer, split: str, epoch: int,
                          scaled_losses: Dict[str, float],
                          unscaled_losses: Dict[str, float],
                          metrics: Dict[str, Any],
                          activation_percent: float,
                          include_scaled: bool,
                          include_classification: bool,
                          include_ntp: bool,
                          include_tik_tok: bool,
                          include_mse: bool,
                          include_cosine: bool,
                          include_ce: bool) -> None:
    """Log the epoch summaries to TensorBoard with optional components toggled.

    Args:
        writer: TensorBoard writer (or ``None`` to skip logging).
        split: Name of the phase (``'train'`` / ``'valid'``) used for metric clarity.
        epoch: Epoch index.
        scaled_losses: Dict containing averaged scaled losses (``total_loss`` etc.).
        unscaled_losses: Dict with averaged unscaled losses (``unscaled_total_loss`` etc.).
        metrics: Dict returned by :func:`summarize_phase_metrics`.
        activation_percent: Codebook activation ratio already multiplied by 100.
        include_scaled: Whether to emit the ``loss/*`` group.
        include_classification: Whether classification metrics are active.
        include_ntp: Whether NTP losses/metrics are active.
        include_tik_tok: Whether TikTok classifier losses/metrics are active.
    """
    if writer is None:
        return

    if include_scaled:
        writer.add_scalar('loss/total', scaled_losses['total_loss'], epoch)
        writer.add_scalar('loss/rec', scaled_losses['rec_loss'], epoch)
        writer.add_scalar('loss/vq', scaled_losses['vq_loss'], epoch)
        if include_mse:
            writer.add_scalar('loss/mse', scaled_losses['mse_loss'], epoch)
        if include_cosine:
            writer.add_scalar('loss/cosine', scaled_losses['cosine_loss'], epoch)
        if include_ce:
            writer.add_scalar('loss/ce', scaled_losses['ce_loss'], epoch)
        if include_ntp:
            writer.add_scalar('loss/ntp', scaled_losses['ntp_loss'], epoch)
        if include_tik_tok:
            writer.add_scalar('loss/tik_tok_padding', scaled_losses['tik_tok_padding_loss'], epoch)
        if include_classification:
            writer.add_scalar('loss/classification', scaled_losses['classification_loss'], epoch)

    writer.add_scalar('unscaled_loss/total', unscaled_losses['unscaled_total_loss'], epoch)
    writer.add_scalar('unscaled_loss/rec_loss', unscaled_losses['unscaled_rec_loss'], epoch)
    writer.add_scalar('unscaled_loss/vq_loss', unscaled_losses['unscaled_vq_loss'], epoch)
    if include_mse:
        writer.add_scalar('unscaled_loss/mse_loss', unscaled_losses['unscaled_mse_loss'], epoch)
    if include_cosine:
        writer.add_scalar('unscaled_loss/cosine_loss', unscaled_losses['unscaled_cosine_loss'], epoch)
    if include_ce:
        writer.add_scalar('unscaled_loss/ce_loss', unscaled_losses['unscaled_ce_loss'], epoch)
    if include_ntp:
        writer.add_scalar('unscaled_loss/ntp_loss', unscaled_losses['unscaled_ntp_loss'], epoch)
    if include_tik_tok:
        writer.add_scalar('unscaled_loss/tik_tok_padding_loss', unscaled_losses['unscaled_tik_tok_padding_loss'], epoch)
    if include_classification:
        writer.add_scalar('unscaled_loss/classification_loss', unscaled_losses['unscaled_classification_loss'], epoch)

    writer.add_scalar('codebook_activation', activation_percent, epoch)

    nmse_value = metrics.get('nmse_median', float('nan'))
    if nmse_value == nmse_value:
        writer.add_scalar('metric/nmse_median', nmse_value, epoch)

    for key in ('nmse_p95', 'nmse_mean', 'nmse_strict_coverage'):
        value = metrics.get(key, float('nan'))
        if value == value:
            writer.add_scalar(f'metric/{key}', value, epoch)

    for key in (
        'recon_ppl_ratio',
        'recon_ppl_pct_delta',
    ):
        value = metrics.get(key, float('nan'))
        if value == value:
            writer.add_scalar(f'metric/{key}', value, epoch)

    if include_classification:
        class_acc = metrics.get('classification_accuracy', float('nan'))
        if class_acc == class_acc:
            writer.add_scalar('metric/classification_accuracy', class_acc, epoch)
        per_class = metrics.get('classification_accuracy_per_class', {})
        for amino_acid, acc in per_class.items():
            if acc == acc:
                writer.add_scalar(f'metric/classification_accuracy_per_class/{amino_acid}', acc, epoch)

    ntp_perplexity = metrics.get('ntp_perplexity', float('nan'))
    if include_ntp and ntp_perplexity == ntp_perplexity:
        writer.add_scalar('metric/ntp_perplexity', ntp_perplexity, epoch)

    tik_tok_accuracy = metrics.get('tik_tok_padding_accuracy', float('nan'))
    kl_value = metrics.get('recon_kl', float('nan'))
    if kl_value == kl_value:
        writer.add_scalar('metric/recon_kl', kl_value, epoch)
    if include_tik_tok and tik_tok_accuracy == tik_tok_accuracy:
        writer.add_scalar('metric/tik_tok_padding_accuracy', tik_tok_accuracy, epoch)
