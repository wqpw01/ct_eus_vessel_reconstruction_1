from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from ct_eus_vessel.image_io import resample_to_reference


def _iterations_for_mm(distance_mm: float, spacing_xyz: tuple[float, float, float]) -> int:
    if distance_mm <= 0:
        return 0
    min_spacing = max(min(spacing_xyz), 1e-6)
    return max(1, int(np.ceil(distance_mm / min_spacing)))


def masks_from_weak_label(
    label_image: sitk.Image,
    reference: sitk.Image,
    *,
    organ_label_ids: set[int],
    vessel_label_ids: set[int],
) -> tuple[None, np.ndarray]:
    label_rs = resample_to_reference(
        label_image,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt16,
    )
    label_arr = sitk.GetArrayFromImage(label_rs)
    soft_ids = organ_label_ids - vessel_label_ids
    return None, np.isin(label_arr, list(soft_ids))


def anchor_mask_from_weak_label(
    label_image: sitk.Image,
    reference: sitk.Image,
    *,
    vessel_label_ids: set[int],
) -> np.ndarray:
    label_rs = resample_to_reference(
        label_image,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt16,
    )
    label_arr = sitk.GetArrayFromImage(label_rs)
    return np.isin(label_arr, list(vessel_label_ids))


def anchor_multilabel_from_weak_label(
    label_image: sitk.Image,
    reference: sitk.Image,
    *,
    arterial_ids: set[int],
    portal_ids: set[int],
    venous_ids: set[int],
) -> np.ndarray:
    label_rs = resample_to_reference(
        label_image,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt16,
    )
    label_arr = sitk.GetArrayFromImage(label_rs)
    out = np.zeros(label_arr.shape, dtype=np.uint8)
    out[np.isin(label_arr, list(arterial_ids))] = 1
    out[np.isin(label_arr, list(portal_ids))] = 2
    out[np.isin(label_arr, list(venous_ids))] = 3
    return out


def bone_like_exclusion(
    image_zyx: np.ndarray,
    *,
    hu_threshold: float,
    dilation_voxels: int,
    preserve_mask: np.ndarray | None,
) -> np.ndarray:
    bone = image_zyx >= hu_threshold
    if dilation_voxels > 0:
        structure = ndi.generate_binary_structure(image_zyx.ndim, 1)
        bone = ndi.binary_dilation(bone, structure=structure, iterations=dilation_voxels)
    if preserve_mask is not None:
        bone = bone & ~preserve_mask
    return bone


def body_region_mask(
    image_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    min_hu: float,
    closing_mm: float,
    dilation_mm: float,
) -> np.ndarray:
    body = np.isfinite(image_zyx) & (image_zyx > min_hu)
    if not body.any():
        return body

    structure = ndi.generate_binary_structure(body.ndim, 1)
    threshold_body = body.copy()
    closing_iterations = _iterations_for_mm(closing_mm, spacing_xyz)
    if closing_iterations > 0:
        body = ndi.binary_closing(body, structure=structure, iterations=closing_iterations)
        if not body.any():
            body = threshold_body
    body = ndi.binary_fill_holes(body)

    labeled, count = ndi.label(body, structure=structure)
    if count == 0:
        return body.astype(bool, copy=False)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    body = labeled == largest_label

    dilation_iterations = _iterations_for_mm(dilation_mm, spacing_xyz)
    if dilation_iterations > 0:
        body = ndi.binary_dilation(body, structure=structure, iterations=dilation_iterations)
    return body.astype(bool, copy=False)
