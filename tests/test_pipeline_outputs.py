import re

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.pipeline import _save_outputs


def _nrrd_header(path) -> str:
    return path.read_bytes().split(b"\n\n", 1)[0].decode("latin1")


def _nrrd_space_origin(path) -> tuple[float, float, float]:
    match = re.search(r"^space origin: \(([^)]+)\)$", _nrrd_header(path), re.MULTILINE)
    assert match is not None
    return tuple(float(value) for value in match.group(1).split(","))


def test_save_outputs_writes_slicer_friendly_ras_nrrd_pair(tmp_path) -> None:
    reference = sitk.Image([5, 4, 3], sitk.sitkInt16)
    reference.SetSpacing((0.5, 0.6, 1.2))
    reference.SetOrigin((10.0, 20.0, 30.0))
    reference.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 1:3, 2:4] = True
    multilabel = mask.astype(np.uint8)
    confidence = mask.astype(np.float32)
    config = {"vessel_extraction": {"bbox_padding_mm": 1}}

    _save_outputs(
        output_dir=tmp_path,
        reference=reference,
        reference_arr=arr,
        arterial_mask=mask,
        portal_mask=mask,
        venous_mask=mask,
        multilabel=multilabel,
        confidence=confidence,
        config=config,
        skip_mesh=True,
    )

    ref_nrrd = sitk.ReadImage(str(tmp_path / "reference_ct.nrrd"))
    seg_nrrd = sitk.ReadImage(str(tmp_path / "vessel_fused_multilabel.nrrd"))
    assert ref_nrrd.GetSize() == seg_nrrd.GetSize()
    assert ref_nrrd.GetSpacing() == seg_nrrd.GetSpacing()
    assert ref_nrrd.GetOrigin() == seg_nrrd.GetOrigin()
    assert ref_nrrd.GetDirection() == seg_nrrd.GetDirection()
    assert ref_nrrd.GetOrigin() == (12.0, 21.8, 30.0)
    assert ref_nrrd.GetDirection() == (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    assert "space: right-anterior-superior" in _nrrd_header(tmp_path / "reference_ct.nrrd")
    assert "space: right-anterior-superior" in _nrrd_header(tmp_path / "vessel_fused_multilabel.nrrd")
    assert _nrrd_space_origin(tmp_path / "reference_ct.nrrd") == (-12.0, -21.8, 30.0)
    assert _nrrd_space_origin(tmp_path / "vessel_fused_multilabel.nrrd") == (-12.0, -21.8, 30.0)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(seg_nrrd), multilabel[:, ::-1, ::-1])

    assert not (tmp_path / "reference_ct.nii.gz").exists()
    assert not (tmp_path / "vessel_fused_multilabel.nii.gz").exists()
    ref_nii = sitk.ReadImage(str(tmp_path / "compat_nifti" / "reference_ct.nii.gz"))
    seg_nii = sitk.ReadImage(str(tmp_path / "compat_nifti" / "vessel_fused_multilabel.nii.gz"))
    assert ref_nii.GetOrigin() == reference.GetOrigin()
    assert ref_nii.GetDirection() == reference.GetDirection()
    assert seg_nii.GetOrigin() == reference.GetOrigin()
    assert seg_nii.GetDirection() == reference.GetDirection()
