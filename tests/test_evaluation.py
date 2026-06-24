from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ct_eus_vessel.evaluation import compare_output_masks


def _write_like(array_zyx: np.ndarray, path: Path, *, spacing=(1.0, 1.0, 1.0)) -> None:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _make_output_pair(tmp_path: Path) -> tuple[Path, Path]:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    ref_mask = np.array(
        [
            [
                [1, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 3, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )
    cand_mask = np.array(
        [
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 3, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )
    candidate_liver = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 5, 5, 0],
                [0, 5, 0, 0],
            ]
        ],
        dtype=np.uint16,
    )

    _write_like(np.zeros_like(ref_mask, dtype=np.int16), reference_dir / "reference_ct.nrrd")
    _write_like(ref_mask, reference_dir / "vessel_fused_multilabel.nrrd")
    _write_like(np.zeros_like(cand_mask, dtype=np.int16), candidate_dir / "reference_ct.nrrd")
    _write_like(cand_mask, candidate_dir / "vessel_fused_multilabel.nrrd")
    _write_like(candidate_liver, candidate_dir / "totalseg" / "roi_subset_multilabel.nii.gz")
    return reference_dir, candidate_dir


def test_compare_output_masks_resamples_candidate_and_reports_liver_metrics(tmp_path: Path) -> None:
    reference_dir, candidate_dir = _make_output_pair(tmp_path)

    metrics = compare_output_masks(reference_dir=reference_dir, candidate_dir=candidate_dir)

    assert metrics["geometry"]["same_size"] is False
    assert metrics["overall"]["reference_voxels"] == 4
    assert metrics["overall"]["candidate_voxels"] == 3
    assert metrics["overall"]["true_positive_voxels"] == 3
    assert metrics["overall"]["recall"] == pytest.approx(0.75)
    assert metrics["overall"]["dice"] == pytest.approx(6 / 7)
    assert metrics["by_label"]["portal"]["reference_voxels"] == 2
    assert metrics["by_label"]["portal"]["recall"] == pytest.approx(0.5)
    assert metrics["liver"]["overall"]["reference_voxels"] == 3
    assert metrics["liver"]["overall"]["candidate_voxels"] == 2
    assert metrics["liver"]["overall"]["recall"] == pytest.approx(2 / 3)
