from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.image_io import resample_to_reference


@dataclass(frozen=True)
class TotalSegPriors:
    hard_exclusion: np.ndarray
    soft_penalty: np.ndarray
    vessel_anchors_by_phase: dict[str, np.ndarray]
    vessel_anchor_any: np.ndarray
    named_masks: dict[str, np.ndarray]


def build_totalsegmentator_command(
    *,
    input_path: Path,
    output_path: Path,
    roi_subset: list[str],
    device: str,
    quiet: bool = False,
) -> list[str]:
    command = [
        "TotalSegmentator",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--ml",
        "--roi_subset",
    ]
    command.extend(roi_subset)
    command.extend(["--device", device])
    if quiet:
        command.append("--quiet")
    return command


def ensure_totalseg_multilabel(
    *,
    reference: sitk.Image,
    output_dir: Path,
    roi_subset: list[str],
    device: str,
    force: bool = False,
) -> Path:
    totalseg_dir = output_dir / "totalseg"
    totalseg_dir.mkdir(parents=True, exist_ok=True)
    input_path = totalseg_dir / "reference_ct_for_totalseg.nii.gz"
    output_path = totalseg_dir / "roi_subset_multilabel.nii.gz"
    if output_path.exists() and not force:
        return output_path
    sitk.WriteImage(reference, str(input_path))
    command = build_totalsegmentator_command(
        input_path=input_path,
        output_path=output_path,
        roi_subset=roi_subset,
        device=device,
        quiet=True,
    )
    subprocess.run(command, check=True)
    return output_path


def resolve_totalseg_label_ids(mask_names: list[str], *, task: str = "total") -> dict[str, int]:
    try:
        from totalsegmentator.map_to_binary import class_map
    except ImportError as exc:
        raise RuntimeError("TotalSegmentator is required to resolve segmentation class names") from exc
    if task not in class_map:
        raise ValueError(f"Unsupported TotalSegmentator task: {task}")
    inverse = {name: int(label_id) for label_id, name in class_map[task].items()}
    missing = [name for name in mask_names if name not in inverse]
    if missing:
        raise ValueError(f"Unknown TotalSegmentator mask names for task {task}: {missing}")
    return {name: inverse[name] for name in mask_names}


def masks_from_totalseg_multilabel(
    label_image: sitk.Image,
    reference: sitk.Image,
    *,
    soft_mask_names: list[str],
    hard_mask_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    priors = priors_from_totalseg_multilabel(
        label_image,
        reference,
        soft_mask_names=soft_mask_names,
        hard_mask_names=hard_mask_names,
        vessel_anchor_mask_names={},
    )
    return priors.hard_exclusion, priors.soft_penalty


def priors_from_totalseg_multilabel(
    label_image: sitk.Image,
    reference: sitk.Image,
    *,
    soft_mask_names: list[str],
    hard_mask_names: list[str],
    vessel_anchor_mask_names: dict[str, list[str]],
) -> TotalSegPriors:
    label_rs = resample_to_reference(
        label_image,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt16,
    )
    label_arr = sitk.GetArrayFromImage(label_rs)
    soft_ids = set(resolve_totalseg_label_ids(soft_mask_names).values())
    hard_ids = set(resolve_totalseg_label_ids(hard_mask_names).values())
    tracked_names = set(soft_mask_names) | set(hard_mask_names)
    for names in vessel_anchor_mask_names.values():
        tracked_names.update(names)
    tracked_ids = resolve_totalseg_label_ids(sorted(tracked_names)) if tracked_names else {}
    soft = np.isin(label_arr, list(soft_ids))
    hard = np.isin(label_arr, list(hard_ids))
    named_masks = {name: label_arr == label_id for name, label_id in tracked_ids.items()}
    anchors_by_phase: dict[str, np.ndarray] = {}
    anchor_any = np.zeros(label_arr.shape, dtype=bool)
    for phase, names in vessel_anchor_mask_names.items():
        ids = set(resolve_totalseg_label_ids(names).values()) if names else set()
        mask = np.isin(label_arr, list(ids)) if ids else np.zeros(label_arr.shape, dtype=bool)
        anchors_by_phase[phase] = mask
        anchor_any |= mask
    return TotalSegPriors(
        hard_exclusion=hard,
        soft_penalty=soft,
        vessel_anchors_by_phase=anchors_by_phase,
        vessel_anchor_any=anchor_any,
        named_masks=named_masks,
    )
