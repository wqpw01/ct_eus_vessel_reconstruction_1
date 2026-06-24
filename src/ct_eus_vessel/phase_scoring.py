from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.phase import PhaseScores


def resample_label_to_image(label_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        label_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )


def _roi_percentile(
    image_arr: np.ndarray,
    label_arr: np.ndarray,
    label_value: int,
    percentile: float,
    min_roi_voxels: int,
) -> float | None:
    voxels = image_arr[label_arr == label_value]
    if voxels.size < min_roi_voxels:
        return None
    return float(np.percentile(voxels, percentile))


def score_phase_image(
    *,
    series_uid: str,
    image: sitk.Image,
    label_image: sitk.Image,
    label_ids: dict[str, int],
    percentile: float,
    min_roi_voxels: int,
) -> PhaseScores:
    resampled_label = resample_label_to_image(label_image, image)
    image_arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    label_arr = sitk.GetArrayFromImage(resampled_label)
    return PhaseScores(
        series_uid=series_uid,
        aorta=_roi_percentile(image_arr, label_arr, label_ids["aorta"], percentile, min_roi_voxels),
        celiac_artery=_roi_percentile(image_arr, label_arr, label_ids["celiac_artery"], percentile, min_roi_voxels),
        portal_vein=_roi_percentile(image_arr, label_arr, label_ids["portal_vein"], percentile, min_roi_voxels),
        ivc=_roi_percentile(image_arr, label_arr, label_ids["ivc"], percentile, min_roi_voxels),
        liver_vein=_roi_percentile(image_arr, label_arr, label_ids["liver_vein"], percentile, min_roi_voxels),
    )
