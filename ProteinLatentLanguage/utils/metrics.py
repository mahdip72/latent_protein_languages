import math
import torch
import torch.distributed as dist
from torchmetrics.metric import Metric


class NMSE(Metric):
    """
    Normalized Mean Squared Error computed per token vector.

    For each valid token t with hidden state x_t and reconstruction x̂_t:
        NMSE_t = ||x_t - x̂_t||² / ||x_t||²

    ``compute`` returns summary statistics over the NMSE_t distribution.
    """

    def __init__(self, dist_sync_on_step: bool = False, eps: float = 1e-8, strict_threshold: float = 0.004):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.eps = eps
        self.strict_threshold = strict_threshold
        self.sync_on_compute = False  # handle distributed sync manually to support variable-length buffers
        self.add_state(
            "nmse_values",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            preds: Reconstructed hidden states, shape (..., dim).
            target: Original hidden states, shape (..., dim).
            mask: Boolean mask selecting valid positions, shape matching leading dims of target.
        """
        if mask.dim() == target.dim():
            mask_tokens = mask[..., 0]
        elif mask.dim() == target.dim() - 1:
            mask_tokens = mask
        else:
            raise ValueError("Mask must have shape matching target without feature dimension.")

        mask_tokens = mask_tokens.bool()
        if mask_tokens.numel() == 0 or not torch.any(mask_tokens):
            return

        # Gather per-token vectors (num_tokens, hidden_dim)
        preds_tokens = preds[mask_tokens]
        target_tokens = target[mask_tokens]

        # Compute per-token NMSE values.
        denom = target_tokens.pow(2).sum(dim=-1).clamp_min(self.eps)
        squared_error = (target_tokens - preds_tokens).pow(2).sum(dim=-1)
        token_nmse = squared_error / denom

        # Accumulate on the same device as the state tensor.
        self.nmse_values = torch.cat(
            (self.nmse_values, token_nmse.detach().to(self.nmse_values.device, dtype=self.nmse_values.dtype))
        )

    def compute(self):
        if self.nmse_values.numel() == 0:
            nan = torch.tensor(float("nan"), device=self.nmse_values.device, dtype=self.nmse_values.dtype)
            return {
                "median": nan,
                "p95": nan,
                "mean": nan,
                "strict_coverage": nan,
                "count": torch.tensor(0, device=self.nmse_values.device, dtype=torch.long),
            }

        values = self.nmse_values

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            count = torch.tensor([values.numel()], device=values.device, dtype=torch.long)
            count_list = [torch.zeros_like(count) for _ in range(world_size)]
            dist.all_gather(count_list, count)
            counts = torch.cat(count_list)
            max_count = int(counts.max().item())

            if max_count == 0:
                values = torch.empty(0, device=values.device, dtype=values.dtype)
            else:
                padded = torch.zeros(max_count, device=values.device, dtype=values.dtype)
                padded[: values.numel()] = values
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                values = torch.cat([
                    tensor[: int(num.item())]
                    for tensor, num in zip(gathered, counts)
                    if num.item() > 0
                ])

        if values.numel() == 0:
            nan = torch.tensor(float("nan"), device=self.nmse_values.device, dtype=self.nmse_values.dtype)
            return {
                "median": nan,
                "p95": nan,
                "mean": nan,
                "strict_coverage": nan,
                "count": torch.tensor(0, device=self.nmse_values.device, dtype=torch.long),
            }

        # Move to CPU for numerically stable quantile/median on large buffers.
        values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
        median_cpu = values_cpu.median()
        mean_cpu = values_cpu.mean()
        coverage_cpu = (values_cpu <= self.strict_threshold).float().mean()

        try:
            p95_cpu = torch.quantile(values_cpu, 0.95)
        except RuntimeError as err:
            if "tensor is too large" not in str(err):
                raise
            # Fallback to kthvalue when quantile is not supported for extremely large tensors.
            n = values_cpu.numel()
            if n == 0:
                p95_cpu = torch.tensor(float("nan"), device=values_cpu.device)
            else:
                k = min(n - 1, max(0, math.ceil(0.95 * (n - 1))))
                # torch.kthvalue expects 1-indexed k
                p95_cpu = torch.kthvalue(values_cpu, k + 1).values

        target_device = self.nmse_values.device
        median = median_cpu.to(device=target_device)
        p95 = p95_cpu.to(device=target_device)
        mean = mean_cpu.to(device=target_device)
        coverage = coverage_cpu.to(device=target_device)
        count = torch.tensor(values_cpu.numel(), device=target_device, dtype=torch.long)

        return {
            "median": median,
            "p95": p95,
            "mean": mean,
            "strict_coverage": coverage,
            "count": count,
        }


def update_ntp_perplexity(ntp_perplexity_metric, ntp_logits, indices, masks=None, ignore_index: int = -100):
    """
    Update a TorchMetrics Perplexity metric from NTP logits and code indices.

    This applies the valid-position mask, constructs next-token labels by shifting
    indices left by one and padding the last position with ignore_index, and then
    calls ntp_perplexity_metric.update(logits, labels).

    Args:
        ntp_perplexity_metric: torchmetrics.text.Perplexity instance or None.
        ntp_logits (Tensor): Shape (B, L, K), unnormalized logits over codebook.
        indices (Tensor): Shape (B, L), integer code indices per position.
        masks (Tensor): Shape (B, L), boolean validity mask.
        ignore_index (int): Label to ignore in perplexity computation (default -100).
    """
    if ntp_perplexity_metric is None or ntp_logits is None or indices is None:
        return

    device = ntp_logits.device

    # Detach and ensure correct dtypes/devices
    logits_detached = ntp_logits.detach()
    indices_long = indices.detach().to(dtype=torch.long, device=device)
    if masks is None:
        masks_bool = torch.ones_like(indices_long, dtype=torch.bool, device=device)
    else:
        masks_bool = masks.detach().to(dtype=torch.bool, device=device)

    B, L, _ = logits_detached.shape
    labels_masked = indices_long.masked_fill(~masks_bool, ignore_index)
    pad_col = torch.full((B, 1), ignore_index, dtype=torch.long, device=device)
    labels = torch.cat([labels_masked[:, 1:], pad_col], dim=1)

    ntp_perplexity_metric.update(logits_detached, labels)


class ReconstructionCrossEntropyDelta(Metric):
    """
    Macro-averaged cross-entropy diagnostics between encoder (base) and decoder (spliced) logits.

    For each sequence we derive:
        * CE_base  = H(p)                  = -∑_i p_i log p_i
        * CE_splice = H(p, q)             = -∑_i p_i log q_i
        * ΔCE      = CE_splice - CE_base
        * RelΔCE   = ΔCE / CE_base        (skipped if CE_base ≤ eps)
        * KL(p‖q)  = ΔCE                  (per-sequence cross entropy difference)
        * PPL_base = exp(CE_base)
        * PPL_splice = exp(CE_splice)
        * PPL_ratio = exp(ΔCE)

    All values are macro-averaged: we compute a per-sequence mean over masked
    tokens, then average across sequences. When no valid tokens are present, we
    surface NaNs for every field.
    """

    def __init__(self, dist_sync_on_step: bool = False, eps: float = 1e-8):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.eps = eps
        self.add_state("sum_delta", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_ppl_ratio", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_sequences", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, base_logits: torch.Tensor, spliced_logits: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            base_logits: Encoder logits (B, L, D) used as the reference distribution.
            spliced_logits: Decoder logits (B, L, D) to evaluate against the reference.
            mask: Boolean mask (B, L) indicating valid token positions.
        """
        if mask.dim() != base_logits.dim() - 1:
            raise ValueError("Mask must have shape (batch, seq_len) matching logits without feature dim.")

        if base_logits.shape != spliced_logits.shape:
            raise ValueError("Base and spliced logits must share the same shape.")

        mask_bool = mask.bool()
        if mask_bool.numel() == 0:
            return

        mask_float = mask_bool.to(dtype=base_logits.dtype)
        valid_counts = mask_float.sum(dim=-1)  # (B,)
        valid_sequences = valid_counts > 0
        if not torch.any(valid_sequences):
            return

        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        base_probs = torch.softmax(base_logits, dim=-1)
        spliced_log_probs = torch.log_softmax(spliced_logits, dim=-1)

        ce_base_token = -(base_probs * base_log_probs).sum(dim=-1)  # (B, L)
        ce_spliced_token = -(base_probs * spliced_log_probs).sum(dim=-1)

        sum_ce_base = (ce_base_token * mask_float).sum(dim=-1)
        sum_ce_spliced = (ce_spliced_token * mask_float).sum(dim=-1)

        mean_ce_base = torch.zeros_like(sum_ce_base)
        mean_ce_spliced = torch.zeros_like(sum_ce_spliced)

        mean_ce_base[valid_sequences] = sum_ce_base[valid_sequences] / valid_counts[valid_sequences]
        mean_ce_spliced[valid_sequences] = sum_ce_spliced[valid_sequences] / valid_counts[valid_sequences]

        delta = mean_ce_spliced[valid_sequences] - mean_ce_base[valid_sequences]

        delta_valid = delta

        self.sum_delta += delta_valid.sum()
        self.sum_ppl_ratio += delta_valid.exp().sum()
        self.count_sequences += valid_sequences.sum().to(self.count_sequences.dtype)

    def compute(self):
        if self.count_sequences == 0:
            nan = torch.tensor(float('nan'), device=self.sum_ce_base.device)
            return {
                "ce_base": nan,
                "ce_spliced": nan,
                "ce_delta": nan,
                "ce_rel_delta": nan,
                "kl": nan,
                "ppl_base": nan,
                "ppl_spliced": nan,
                "ppl_ratio": nan,
            }

        ce_delta_macro = self.sum_delta / self.count_sequences
        ppl_ratio_macro = self.sum_ppl_ratio / self.count_sequences

        return {
            "ce_delta": ce_delta_macro,
            "kl": ce_delta_macro,
            "ppl_ratio": ppl_ratio_macro,
        }
