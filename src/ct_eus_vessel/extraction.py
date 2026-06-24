from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi


@dataclass(frozen=True)
class CandidateResult:
    mask: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class FusedResult:
    multilabel: np.ndarray
    confidence: np.ndarray


def remove_small_components(
    mask: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    min_volume_mm3: float,
) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool, copy=True)
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    min_voxels = max(1, int(np.ceil(min_volume_mm3 / voxel_volume)))
    sizes = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(sizes >= min_voxels)
    keep_labels = keep_labels[keep_labels != 0]
    return np.isin(labeled, keep_labels)


def keep_components_near_anchors(mask: np.ndarray, anchors: np.ndarray, *, dilation_voxels: int) -> np.ndarray:
    if mask.sum() == 0 or anchors.sum() == 0:
        return mask.astype(bool, copy=True)
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    expanded_anchors = ndi.binary_dilation(anchors, structure=structure, iterations=dilation_voxels)
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool, copy=True)
    touching_labels = np.unique(labeled[expanded_anchors & (labeled > 0)])
    return np.isin(labeled, touching_labels)


def keep_mask_within_anchor_distance(
    mask: np.ndarray,
    anchors: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    max_distance_mm: float,
) -> np.ndarray:
    if mask.sum() == 0 or anchors.sum() == 0:
        return mask.astype(bool, copy=True)
    sampling_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    distance = ndi.distance_transform_edt(~anchors, sampling=sampling_zyx)
    return mask & (distance <= max_distance_mm)


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.float32)
    min_value = float(arr[valid].min())
    max_value = float(arr[valid].max())
    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - min_value) / (max_value - min_value), 0.0, 1.0).astype(np.float32)


def compute_frangi_vesselness(
    image_zyx: np.ndarray,
    *,
    sigmas_voxels: list[float],
) -> np.ndarray:
    clipped = np.clip(image_zyx.astype(np.float32, copy=False), -200, 500)
    # Bright contrast-filled vessels are ridges on CT, so black_ridges=False.
    return normalize01(frangi(clipped, sigmas=sigmas_voxels, black_ridges=False))


def compute_slice_frangi_vesselness(
    image_zyx: np.ndarray,
    *,
    sigmas_voxels: list[float],
) -> np.ndarray:
    clipped = np.clip(image_zyx.astype(np.float32, copy=False), -200, 500)
    out = np.zeros(clipped.shape, dtype=np.float32)
    for z_index in range(clipped.shape[0]):
        out[z_index] = frangi(clipped[z_index], sigmas=sigmas_voxels, black_ridges=False).astype(np.float32)
    return normalize01(out)


def extract_vessel_candidate(
    image_zyx: np.ndarray,
    *,
    vesselness: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    hu_low: float,
    hu_high: float,
    vesselness_min: float,
    hard_exclusion_mask: np.ndarray | None,
    soft_penalty_mask: np.ndarray | None,
    min_component_volume_mm3: float,
) -> CandidateResult:
    hu_mask = (image_zyx >= hu_low) & (image_zyx <= hu_high)
    vesselness_norm = normalize01(vesselness)
    confidence = vesselness_norm * hu_mask.astype(np.float32)
    if soft_penalty_mask is not None:
        confidence = confidence.copy()
        confidence[soft_penalty_mask] *= 0.65
    candidate = hu_mask & (vesselness_norm >= vesselness_min)
    if hard_exclusion_mask is not None:
        candidate &= ~hard_exclusion_mask
        confidence = confidence.copy()
        confidence[hard_exclusion_mask] = 0.0
    candidate = remove_small_components(
        candidate,
        spacing_xyz=spacing_xyz,
        min_volume_mm3=min_component_volume_mm3,
    )
    confidence = confidence.astype(np.float32, copy=False)
    confidence[~candidate] = 0.0
    return CandidateResult(mask=candidate, confidence=confidence)


def fuse_phase_candidates(
    *,
    arterial_mask: np.ndarray,
    portal_mask: np.ndarray,
    venous_mask: np.ndarray,
    confidence_maps: dict[str, np.ndarray],
) -> FusedResult:
    stacked_conf = np.stack(
        [
            np.where(arterial_mask, confidence_maps["arterial"], 0.0),
            np.where(portal_mask, confidence_maps["portal"], 0.0),
            np.where(venous_mask, confidence_maps["venous"], 0.0),
        ],
        axis=0,
    )
    best = np.argmax(stacked_conf, axis=0).astype(np.uint8) + 1
    confidence = np.max(stacked_conf, axis=0).astype(np.float32)
    multilabel = np.where(confidence > 0, best, 0).astype(np.uint8)
    return FusedResult(multilabel=multilabel, confidence=confidence)
