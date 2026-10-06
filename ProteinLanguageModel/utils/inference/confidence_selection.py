"""Paper confidence scoring/selection extracted without benchmark or refolding code."""
import math
from typing import Any

def _parse_metric_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split():
        try:
            val = float(token)
        except ValueError:
            continue
        if math.isnan(val):
            continue
        values.append(val)
    return values


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float('nan')
    return sum(values) / len(values)


DEEPCONF_SCORE_FIELDS = [
    "deepconf_mean_entropy_raw",
    "deepconf_tail_entropy_raw",
    "deepconf_worst_group_entropy_raw",
    "deepconf_top10pct_group_entropy_raw",
]


def _resolve_deepconf_selection_config(infer_cfg: Any) -> dict[str, Any]:
    cfg = infer_cfg.get("deepconf_selection", None)
    if cfg is None:
        cfg = infer_cfg.get("deepconf_selectoin", None)
    if cfg is None:
        legacy_cfg = infer_cfg.get("entropy_selection", None)
        if isinstance(legacy_cfg, bool):
            cfg = {
                "enabled": legacy_cfg,
                "score": "mean_structure_entropy_raw",
            }
        elif legacy_cfg:
            cfg = {
                "enabled": bool(legacy_cfg.get("enabled", False)),
                "score": "mean_structure_entropy_raw",
            }
        else:
            cfg = {}
    elif isinstance(cfg, bool):
        cfg = {"enabled": cfg}
    else:
        cfg = dict(cfg)

    enabled = bool(cfg.get("enabled", False))
    score = str(cfg.get("score", "deepconf_top10pct_group_entropy_raw"))
    group_size = max(1, int(cfg.get("group_size", 32)))
    group_stride = max(1, int(cfg.get("group_stride", 1)))
    bottom_percent = float(cfg.get("bottom_percent", 0.10))
    if not 0.0 < bottom_percent <= 1.0:
        if enabled:
            raise ValueError("deepconf_selection.bottom_percent must be in (0, 1].")
        bottom_percent = 0.10
    tail_size_value = cfg.get("tail_size", 64)
    tail_size = None if tail_size_value is None else max(1, int(tail_size_value))
    consensus_cfg = cfg.get("consensus", {}) or {}
    if isinstance(consensus_cfg, bool):
        consensus_cfg = {"enabled": consensus_cfg}
    else:
        consensus_cfg = dict(consensus_cfg)
    keep_percent = float(consensus_cfg.get("keep_percent", cfg.get("keep_percent", 0.50)))
    if not 0.0 < keep_percent <= 1.0:
        if enabled and bool(consensus_cfg.get("enabled", False)):
            raise ValueError("deepconf_selection.consensus.keep_percent must be in (0, 1].")
        keep_percent = 0.50
    min_keep = max(1, int(consensus_cfg.get("min_keep", 2)))

    return {
        "enabled": enabled,
        "score": score,
        "group_size": group_size,
        "group_stride": group_stride,
        "bottom_percent": bottom_percent,
        "tail_size": tail_size,
        "consensus": {
            "enabled": bool(consensus_cfg.get("enabled", False)),
            "keep_percent": keep_percent,
            "min_keep": min_keep,
            "metric": str(consensus_cfg.get("metric", "token_similarity")),
        },
    }


def _sliding_window_means(values: list[float], window_size: int, stride: int) -> list[float]:
    if not values:
        return []
    window_size = min(max(1, window_size), len(values))
    stride = max(1, stride)
    if window_size == 1:
        return values[::stride]
    cumsum = [0.0]
    for value in values:
        cumsum.append(cumsum[-1] + value)
    means = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        means.append((cumsum[end] - cumsum[start]) / window_size)
    return means


def compute_deepconf_entropy_scores(
    entropy_values: list[float],
    deepconf_cfg: dict[str, Any],
) -> dict[str, float]:
    finite_values = []
    for value in entropy_values:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(numeric_value):
            finite_values.append(numeric_value)
    if not finite_values:
        return {field: float('nan') for field in DEEPCONF_SCORE_FIELDS}

    tail_size = deepconf_cfg.get("tail_size")
    tail_values = finite_values[-tail_size:] if tail_size else finite_values
    group_means = _sliding_window_means(
        finite_values,
        int(deepconf_cfg.get("group_size", 32)),
        int(deepconf_cfg.get("group_stride", 1)),
    )
    if not group_means:
        group_means = finite_values
    bottom_percent = float(deepconf_cfg.get("bottom_percent", 0.10))
    worst_count = max(1, math.ceil(len(group_means) * bottom_percent))
    worst_group_means = sorted(group_means, reverse=True)[:worst_count]

    return {
        "deepconf_mean_entropy_raw": _mean_or_nan(finite_values),
        "deepconf_tail_entropy_raw": _mean_or_nan(tail_values),
        "deepconf_worst_group_entropy_raw": max(group_means),
        "deepconf_top10pct_group_entropy_raw": _mean_or_nan(worst_group_means),
    }


def _token_sequence_similarity(
    left: list[int] | None,
    right: list[int] | None,
    metric: str,
) -> float:
    if not left or not right:
        return float('nan')
    metric = metric.lower()
    if metric in {"token_jaccard", "jaccard"}:
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        return len(left_set & right_set) / len(union) if union else float('nan')

    overlap = min(len(left), len(right))
    if overlap == 0:
        return float('nan')
    matches = sum(1 for idx in range(overlap) if left[idx] == right[idx])
    return matches / max(len(left), len(right))


def _select_deepconf_candidate(
    candidates: list[dict[str, Any]],
    deepconf_metric: str,
    deepconf_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    scored_candidates = [
        item for item in candidates
        if not math.isnan(item.get(deepconf_metric, float('nan')))
    ]
    if not scored_candidates:
        return None

    scored_candidates = sorted(
        scored_candidates,
        key=lambda item: item[deepconf_metric],
    )
    consensus_cfg = deepconf_cfg.get("consensus", {}) or {}
    if not bool(consensus_cfg.get("enabled", False)):
        return scored_candidates[0]

    candidates_with_tokens = [
        item for item in scored_candidates
        if item.get("_structure_tokens")
    ]
    if not candidates_with_tokens:
        return scored_candidates[0]

    keep_count = math.ceil(len(candidates_with_tokens) * float(consensus_cfg.get("keep_percent", 0.50)))
    keep_count = max(int(consensus_cfg.get("min_keep", 2)), keep_count)
    keep_count = min(len(candidates_with_tokens), keep_count)
    retained = candidates_with_tokens[:keep_count]
    if len(retained) == 1:
        return retained[0]

    similarity_metric = str(consensus_cfg.get("metric", "token_similarity"))
    best_candidate = None
    best_key = None
    for candidate in retained:
        similarities = []
        for other in retained:
            if other is candidate:
                continue
            similarity = _token_sequence_similarity(
                candidate.get("_structure_tokens"),
                other.get("_structure_tokens"),
                similarity_metric,
            )
            if not math.isnan(similarity):
                similarities.append(similarity)
        avg_similarity = _mean_or_nan(similarities)
        selection_key = (
            avg_similarity if not math.isnan(avg_similarity) else -1.0,
            -candidate[deepconf_metric],
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_candidate = candidate
    return best_candidate or retained[0]
