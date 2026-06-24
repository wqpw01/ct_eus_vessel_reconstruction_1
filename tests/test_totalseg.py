from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.totalseg import (
    build_totalsegmentator_command,
    ensure_totalseg_multilabel,
    masks_from_totalseg_multilabel,
    priors_from_totalseg_multilabel,
    resolve_totalseg_label_ids,
)


def test_build_totalsegmentator_command_uses_roi_subset_and_multilabel(tmp_path: Path) -> None:
    command = build_totalsegmentator_command(
        input_path=tmp_path / "ct.nii.gz",
        output_path=tmp_path / "totalseg.nii.gz",
        roi_subset=["liver", "spleen", "vertebrae_L1"],
        device="gpu",
    )

    assert command[:2] == ["TotalSegmentator", "-i"]
    assert "--roi_subset" in command
    assert "vertebrae_L1" in command
    assert "--ml" in command
    assert command[-2:] == ["--device", "gpu"]


def test_build_totalsegmentator_command_can_run_quietly(tmp_path: Path) -> None:
    command = build_totalsegmentator_command(
        input_path=tmp_path / "ct.nii.gz",
        output_path=tmp_path / "totalseg.nii.gz",
        roi_subset=["liver"],
        device="gpu",
        quiet=True,
    )

    assert "--quiet" in command


def test_resolve_totalseg_label_ids_uses_total_class_map() -> None:
    ids = resolve_totalseg_label_ids(["liver", "vertebrae_L1"])

    assert ids["liver"] == 5
    assert ids["vertebrae_L1"] == 31


def test_masks_from_totalseg_multilabel_builds_soft_and_hard_masks() -> None:
    label = sitk.GetImageFromArray(np.array([[[5, 31], [0, 19]]], dtype=np.uint16))
    reference = sitk.Image(label)

    hard, soft = masks_from_totalseg_multilabel(
        label,
        reference,
        soft_mask_names=["liver", "duodenum"],
        hard_mask_names=["vertebrae_L1"],
    )

    assert soft[0, 0, 0]
    assert soft[0, 1, 1]
    assert not soft[0, 0, 1]
    assert hard[0, 0, 1]
    assert not hard[0, 0, 0]


def test_priors_from_totalseg_multilabel_builds_phase_vessel_anchors() -> None:
    label = sitk.GetImageFromArray(
        np.array(
            [
                [
                    [5, 31, 52],
                    [64, 63, 0],
                ]
            ],
            dtype=np.uint16,
        )
    )
    reference = sitk.Image(label)

    priors = priors_from_totalseg_multilabel(
        label,
        reference,
        soft_mask_names=["liver"],
        hard_mask_names=["vertebrae_L1"],
        vessel_anchor_mask_names={
            "arterial": ["aorta"],
            "portal": ["portal_vein_and_splenic_vein"],
            "venous": ["inferior_vena_cava"],
        },
    )

    assert priors.soft_penalty[0, 0, 0]
    assert priors.hard_exclusion[0, 0, 1]
    assert priors.vessel_anchors_by_phase["arterial"][0, 0, 2]
    assert priors.vessel_anchors_by_phase["portal"][0, 1, 0]
    assert priors.vessel_anchors_by_phase["venous"][0, 1, 1]
    assert priors.vessel_anchor_any.sum() == 3


def test_ensure_totalseg_multilabel_reuses_existing_cache(tmp_path: Path, monkeypatch) -> None:
    reference = sitk.Image([2, 2, 1], sitk.sitkInt16)
    cached = tmp_path / "totalseg" / "roi_subset_multilabel.nii.gz"
    cached.parent.mkdir()
    cached.write_bytes(b"cached")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("TotalSegmentator should not run when cache exists")

    monkeypatch.setattr("subprocess.run", fail_run)

    result = ensure_totalseg_multilabel(
        reference=reference,
        output_dir=tmp_path,
        roi_subset=["liver"],
        device="gpu",
        force=False,
    )

    assert result == cached
