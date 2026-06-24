from __future__ import annotations

import numpy as np

from ct_eus_vessel.phase import PhaseScores


def _valid(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def phase_hu_window(
    scores: PhaseScores,
    *,
    phase: str,
    default_low: float,
    default_high: float,
) -> tuple[int, int]:
    if phase == "arterial":
        anchors = _valid([scores.aorta, scores.celiac_artery])
    elif phase == "portal":
        anchors = _valid([scores.portal_vein])
    elif phase == "venous":
        anchors = _valid([scores.liver_vein, scores.ivc])
    else:
        anchors = []
    if not anchors:
        return int(default_low), int(default_high)
    peak = max(anchors)
    low = max(default_low, min(140.0, peak * 0.5))
    high = max(default_high, min(700.0, peak + 128.0))
    return int(round(low)), int(round(high))


def image_hu_window(
    image_zyx: np.ndarray,
    *,
    default_low: float,
    default_high: float,
    hard_exclusion_mask: np.ndarray | None,
    soft_penalty_mask: np.ndarray | None,
    min_voxels: int = 512,
    upper_limit: float = 700.0,
    percentile: float = 95.0,
) -> tuple[int, int]:
    arr = image_zyx.astype(np.float32, copy=False)
    candidate = np.isfinite(arr) & (arr >= default_low) & (arr <= upper_limit)
    if hard_exclusion_mask is not None:
        candidate &= ~hard_exclusion_mask
    if soft_penalty_mask is not None:
        candidate &= ~soft_penalty_mask
    values = arr[candidate]
    if values.size < min_voxels:
        return int(default_low), int(default_high)
    peak = float(np.percentile(values, percentile))
    low = max(default_low, min(140.0, peak * 0.5))
    high = max(default_high, min(upper_limit, peak + 128.0))
    return int(round(low)), int(round(high))


def anchor_hu_window(
    image_zyx: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    default_low: float,
    default_high: float,
    min_voxels: int = 64,
    upper_limit: float = 700.0,
    percentile: float = 75.0,
) -> tuple[int, int]:
    arr = image_zyx.astype(np.float32, copy=False)
    values = arr[np.isfinite(arr) & anchor_mask & (arr >= default_low) & (arr <= upper_limit)]
    if values.size < min_voxels:
        return int(default_low), int(default_high)
    peak = float(np.percentile(values, percentile))
    low = max(default_low, min(140.0, peak * 0.5))
    high = max(default_high, min(upper_limit, peak + 128.0))
    return int(round(low)), int(round(high))
