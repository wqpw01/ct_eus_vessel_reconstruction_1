from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.image_io import resample_to_reference
from ct_eus_vessel.totalseg import resolve_totalseg_label_ids


LABEL_NAMES = {
    1: "arterial",
    2: "portal",
    3: "venous",
}


def _read_image(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return sitk.ReadImage(str(path))


def _resampled_array(path: Path, reference: sitk.Image, *, pixel_id: int) -> np.ndarray:
    image = _read_image(path)
    resampled = resample_to_reference(
        image,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=pixel_id,
    )
    return sitk.GetArrayFromImage(resampled)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _mask_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = candidate.astype(bool, copy=False)
    reference = reference.astype(bool, copy=False)
    tp = int((candidate & reference).sum())
    fp = int((candidate & ~reference).sum())
    fn = int((~candidate & reference).sum())
    candidate_voxels = int(candidate.sum())
    reference_voxels = int(reference.sum())
    dice_denominator = candidate_voxels + reference_voxels
    return {
        "reference_voxels": reference_voxels,
        "candidate_voxels": candidate_voxels,
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "dice": _ratio(2 * tp, dice_denominator),
    }


def _geometry(reference: sitk.Image, candidate: sitk.Image) -> dict[str, Any]:
    return {
        "reference_size": tuple(int(v) for v in reference.GetSize()),
        "candidate_size": tuple(int(v) for v in candidate.GetSize()),
        "reference_spacing": tuple(float(v) for v in reference.GetSpacing()),
        "candidate_spacing": tuple(float(v) for v in candidate.GetSpacing()),
        "reference_origin": tuple(float(v) for v in reference.GetOrigin()),
        "candidate_origin": tuple(float(v) for v in candidate.GetOrigin()),
        "reference_direction": tuple(float(v) for v in reference.GetDirection()),
        "candidate_direction": tuple(float(v) for v in candidate.GetDirection()),
        "same_size": reference.GetSize() == candidate.GetSize(),
        "same_spacing": reference.GetSpacing() == candidate.GetSpacing(),
        "same_origin": reference.GetOrigin() == candidate.GetOrigin(),
        "same_direction": reference.GetDirection() == candidate.GetDirection(),
    }


def _liver_mask(candidate_dir: Path, reference: sitk.Image) -> np.ndarray:
    path = candidate_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
    if not path.exists():
        return np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=bool)
    liver_id = resolve_totalseg_label_ids(["liver"])["liver"]
    label = _resampled_array(path, reference, pixel_id=sitk.sitkUInt16)
    return label == liver_id


def compare_output_masks(*, reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_dir = Path(reference_dir)
    candidate_dir = Path(candidate_dir)
    reference_ct = _read_image(reference_dir / "reference_ct.nrrd")
    candidate_ct = _read_image(candidate_dir / "reference_ct.nrrd")
    reference_multilabel = _resampled_array(
        reference_dir / "vessel_fused_multilabel.nrrd",
        reference_ct,
        pixel_id=sitk.sitkUInt8,
    ).astype(np.uint8, copy=False)
    candidate_multilabel = _resampled_array(
        candidate_dir / "vessel_fused_multilabel.nrrd",
        reference_ct,
        pixel_id=sitk.sitkUInt8,
    ).astype(np.uint8, copy=False)
    liver = _liver_mask(candidate_dir, reference_ct)

    by_label = {
        name: _mask_metrics(candidate_multilabel == label_id, reference_multilabel == label_id)
        for label_id, name in LABEL_NAMES.items()
    }
    liver_by_label = {
        name: _mask_metrics((candidate_multilabel == label_id) & liver, (reference_multilabel == label_id) & liver)
        for label_id, name in LABEL_NAMES.items()
    }
    return {
        "geometry": _geometry(reference_ct, candidate_ct),
        "overall": _mask_metrics(candidate_multilabel > 0, reference_multilabel > 0),
        "by_label": by_label,
        "liver": {
            "overall": _mask_metrics((candidate_multilabel > 0) & liver, (reference_multilabel > 0) & liver),
            "by_label": liver_by_label,
        },
    }
