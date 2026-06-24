from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class IntrahepaticRecoveryResult:
    mask: np.ndarray
    confidence: np.ndarray
    metrics: dict[str, int]


def _iterations_for_mm(distance_mm: float, spacing_xyz: tuple[float, float, float]) -> int:
    if distance_mm <= 0:
        return 0
    min_spacing = max(min(spacing_xyz), 1e-6)
    return max(1, int(np.ceil(distance_mm / min_spacing)))


def _sigma_zyx(sigma_mm: float, spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    if sigma_mm <= 0:
        return (0.0, 0.0, 0.0)
    return (
        float(sigma_mm) / max(float(spacing_xyz[2]), 1e-6),
        float(sigma_mm) / max(float(spacing_xyz[1]), 1e-6),
        float(sigma_mm) / max(float(spacing_xyz[0]), 1e-6),
    )


def _component_elongation(coords_zyx: np.ndarray) -> float:
    extent = coords_zyx.max(axis=0) - coords_zyx.min(axis=0) + 1
    positive = extent[extent > 0]
    return float(positive.max() / max(positive.min(), 1))


def _empty_like(image_zyx: np.ndarray) -> IntrahepaticRecoveryResult:
    mask = np.zeros(image_zyx.shape, dtype=bool)
    confidence = np.zeros(image_zyx.shape, dtype=np.float32)
    return IntrahepaticRecoveryResult(
        mask=mask,
        confidence=confidence,
        metrics={
            "candidate_voxels": 0,
            "kept_voxels": 0,
            "kept_components": 0,
            "rejected_components": 0,
        },
    )


def _masked_gaussian_background(image_zyx: np.ndarray, mask: np.ndarray, sigma_zyx: tuple[float, float, float]) -> np.ndarray:
    weights = mask.astype(np.float32, copy=False)
    numerator = ndi.gaussian_filter(image_zyx * weights, sigma=sigma_zyx)
    denominator = ndi.gaussian_filter(weights, sigma=sigma_zyx)
    return np.divide(
        numerator,
        denominator,
        out=image_zyx.astype(np.float32, copy=True),
        where=denominator > 1e-3,
    )


def recover_intrahepatic_vessels(
    image_zyx: np.ndarray,
    *,
    vesselness: np.ndarray,
    liver_mask: np.ndarray,
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    anchor_mask: np.ndarray,
    hu_window: tuple[int, int],
    spacing_xyz: tuple[float, float, float],
    config: dict[str, Any],
    phase_name: str | None = None,
) -> IntrahepaticRecoveryResult:
    if not config.get("enabled", True):
        return _empty_like(image_zyx)
    liver = liver_mask.astype(bool, copy=False)
    if not liver.any():
        return _empty_like(image_zyx)

    arr = image_zyx.astype(np.float32, copy=False)
    local_sigma = _sigma_zyx(float(config.get("local_background_sigma_mm", 6.0)), spacing_xyz)
    background = _masked_gaussian_background(arr, liver, local_sigma)
    local_contrast = arr - background

    low, high = hu_window
    hu_low = max(80.0, float(low) - float(config.get("hu_low_margin", 20.0)))
    hu_high = min(float(high), float(config.get("hu_high_cap", 350.0)))
    relaxed_vesselness = float(config.get("relaxed_vesselness_min", 0.005))
    phase_contrast = config.get("phase_local_contrast_min_hu", {})
    if phase_name is not None and isinstance(phase_contrast, dict) and phase_name in phase_contrast:
        contrast_min = float(phase_contrast[phase_name])
    else:
        contrast_min = float(config.get("local_contrast_min_hu", 8.0))
    allowed = liver & body_mask.astype(bool, copy=False) & ~hard_exclusion_mask.astype(bool, copy=False)
    hu_mask = (arr >= hu_low) & (arr <= hu_high)
    response = (vesselness >= relaxed_vesselness) | (local_contrast >= contrast_min)
    candidate = allowed & hu_mask & response

    structure = ndi.generate_binary_structure(candidate.ndim, 1)
    closing_iterations = _iterations_for_mm(float(config.get("closing_mm", 1.5)), spacing_xyz)
    if closing_iterations > 0 and candidate.any():
        closed = ndi.binary_closing(candidate, structure=structure, iterations=closing_iterations)
        closed &= allowed
        if closed.any():
            candidate = closed
    candidate_voxels = int(candidate.sum())
    if candidate_voxels == 0:
        return _empty_like(image_zyx)

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    min_voxels = max(1, int(np.ceil(float(config.get("min_component_volume_mm3", 4.0)) / voxel_volume)))
    max_voxels = max(1, int(float(config.get("max_component_liver_fraction", 0.08)) * int(liver.sum())))
    anchor_iterations = _iterations_for_mm(float(config.get("anchor_dilation_mm", 10.0)), spacing_xyz)
    expanded_anchor = anchor_mask.astype(bool, copy=False)
    if anchor_iterations > 0 and expanded_anchor.any():
        expanded_anchor = ndi.binary_dilation(expanded_anchor, structure=structure, iterations=anchor_iterations)
    min_elongation = float(config.get("component_min_elongation", 2.0))

    labeled, count = ndi.label(candidate, structure=structure)
    keep = np.zeros(candidate.shape, dtype=bool)
    kept_components = 0
    rejected_components = 0
    component_slices = ndi.find_objects(labeled, max_label=count)
    for label_id, bbox in enumerate(component_slices, start=1):
        if bbox is None:
            continue
        labeled_view = labeled[bbox]
        component_view = labeled_view == label_id
        component_voxels = int(component_view.sum())
        if component_voxels < min_voxels or component_voxels > max_voxels:
            rejected_components += 1
            continue
        anchor_view = expanded_anchor[bbox]
        touches_anchor = bool((component_view & anchor_view).any())
        coords = np.argwhere(component_view)
        elongated = _component_elongation(coords) >= min_elongation
        mean_contrast = float(local_contrast[bbox][component_view].mean()) if component_voxels else 0.0
        if touches_anchor or (elongated and mean_contrast >= contrast_min):
            keep[bbox] |= component_view
            kept_components += 1
        else:
            rejected_components += 1

    confidence = np.zeros(arr.shape, dtype=np.float32)
    if keep.any():
        contrast_score = np.clip(local_contrast / 40.0, 0.0, 1.0)
        vesselness_score = np.clip(vesselness / 0.02, 0.0, 1.0)
        confidence = (0.35 + 0.4 * contrast_score + 0.25 * vesselness_score).astype(np.float32)
        confidence[~keep] = 0.0
    return IntrahepaticRecoveryResult(
        mask=keep,
        confidence=confidence,
        metrics={
            "candidate_voxels": candidate_voxels,
            "kept_voxels": int(keep.sum()),
            "kept_components": int(kept_components),
            "rejected_components": int(rejected_components),
        },
    )
