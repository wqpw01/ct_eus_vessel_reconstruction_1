from pathlib import Path

import numpy as np
import SimpleITK as sitk

import ct_eus_vessel.pipeline as pipeline
from ct_eus_vessel.series import SeriesInfo
from ct_eus_vessel.totalseg import resolve_totalseg_label_ids


def _image(value: int) -> sitk.Image:
    arr = np.full((4, 8, 8), value, dtype=np.int16)
    arr[:, 3:5, 3:5] = 180
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def _table_case_image() -> sitk.Image:
    arr = np.full((4, 16, 16), -1000, dtype=np.int16)
    arr[:, 2:10, 2:10] = 40
    arr[:, 5:7, 5:7] = 180
    arr[:, 14, 1:15] = 300
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def _liver_branch_image() -> sitk.Image:
    arr = np.full((4, 16, 16), -1000, dtype=np.int16)
    arr[:, 3:13, 3:13] = 110
    arr[:, 4, 4:12] = 170
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def _image_with_overrides(
    default_value: int,
    overrides: dict[tuple[int, int, int], int],
    *,
    shape_zyx: tuple[int, int, int] = (4, 8, 8),
) -> sitk.Image:
    arr = np.full(shape_zyx, default_value, dtype=np.int16)
    for index, value in overrides.items():
        arr[index] = value
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def test_run_pipeline_without_label_uses_auto_totalseg_priors(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.172466",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="163714.396",
        ),
        SeriesInfo(
            series_uid="uid.182466",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="163738.396",
        ),
        SeriesInfo(
            series_uid="uid.14300",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="163803.145",
        ),
    ]
    images = {
        "uid.172466": _image(90),
        "uid.182466": _image(100),
        "uid.14300": _image(110),
    }

    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label = sitk.Image(reference.GetSize(), sitk.sitkUInt16)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    input_path = tmp_path / "dicom"
    input_path.mkdir()
    (tmp_path / "pseudo_label-.nii").write_bytes(b"not a valid image")

    summary = pipeline.run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=None,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )

    assert summary["guidance_source"] == "auto_totalseg_priors"
    assert summary["phase_mapping"].arterial_uid == "uid.172466"
    assert summary["phase_mapping"].portal_uid == "uid.182466"
    assert summary["phase_mapping"].venous_uid == "uid.14300"


def test_run_pipeline_without_label_uses_body_mask_and_totalseg_vessel_anchors(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    images = {item.series_uid: _table_case_image() for item in candidates}

    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[:, 5:7, 5:7] = 52
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=None,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[:, 5:7, 5:7].any()
    assert not fused[:, 14, 1:15].any()
    assert summary["anchor_source"] == "totalseg_vessel_priors"
    assert summary["quality_metrics"]["outside_body_voxels"] == 0


def test_run_pipeline_without_label_recovers_low_frangi_intrahepatic_branch(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    images = {item.series_uid: _liver_branch_image() for item in candidates}

    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)
    monkeypatch.setattr(pipeline, "compute_slice_frangi_vesselness", lambda arr, sigmas_voxels: np.zeros(arr.shape, dtype=np.float32))

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[:, 3:13, 3:13] = 5
        label_arr[:, 12, 12] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=None,
        skip_frangi=False,
        skip_mesh=True,
        vesselness_mode="slice-frangi",
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[:, 4, 5:12].any()
    assert summary["quality_metrics"]["intrahepatic_recovery_voxels"] > 0


def test_run_pipeline_without_label_injects_totalseg_vessel_anchors_into_output(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    images = {item.series_uid: _image(100) for item in candidates}
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def empty_phase_candidate(image, reference, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        return (
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=np.float32),
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", empty_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1, 2, 2] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=None,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[1, 2, 2] == 2
    assert summary["quality_metrics"]["totalseg_anchor_output_voxels"] == 1
    assert summary["quality_metrics"]["totalseg_anchor_output_by_phase"]["portal"]["injected_voxels"] == 1


def test_run_pipeline_without_label_prunes_low_confidence_liver_surface_recovery(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    images = {item.series_uid: _image(100) for item in candidates}
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            low_surface = (1, 1, 1)
            high_surface = (1, 1, 2)
            low_deep = (2, 3, 3)
            for index, value in [(low_surface, 0.5), (high_surface, 0.9), (low_deep, 0.5)]:
                mask[index] = True
                confidence[index] = value
                recovery[index] = True
        return (
            mask,
            confidence,
            {"candidate_voxels": int(recovery.sum()), "kept_voxels": int(recovery.sum()), "kept_components": int(recovery.any()), "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:4, 1:6, 1:6] = 5
        label_arr[0, 0, 0] = 52
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "surface_prune.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_depth_mm: 1.1\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[1, 1, 1] == 0
    assert fused[1, 1, 2] == 3
    assert fused[2, 3, 3] == 3
    assert summary["quality_metrics"]["intrahepatic_surface_pruned_voxels"] == 1


def test_run_pipeline_v3_relabels_portal_like_venous_and_cleans_final_liver_surface(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    relabel_voxel = (2, 3, 3)
    cleanup_voxel = (1, 1, 1)
    retained_venous = (1, 1, 2)
    images = {
        "uid.arterial": _image_with_overrides(100, {}),
        "uid.portal": _image_with_overrides(
            100,
            {
                relabel_voxel: 180,
                cleanup_voxel: 120,
                retained_venous: 125,
            },
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                relabel_voxel: 130,
                cleanup_voxel: 120,
                retained_venous: 120,
            },
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        if phase_name == "venous":
            for index, value in [(relabel_voxel, 0.7), (cleanup_voxel, 0.5), (retained_venous, 0.9)]:
                mask[index] = True
                confidence[index] = value
        return (
            mask,
            confidence,
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:4, 1:6, 1:6] = 5
        label_arr[0, 3, 3] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  body_closing_mm: 0\n"
        "  body_dilation_mm: 0\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    confidence_min: 0.78\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[relabel_voxel] == 2
    assert fused[cleanup_voxel] == 0
    assert fused[retained_venous] == 3
    assert summary["quality_metrics"]["portal_relabel_voxels"] == 1
    assert summary["quality_metrics"]["final_liver_surface_cleanup_voxels"] == 1
    assert summary["quality_metrics"]["outside_body_voxels"] == 0
    assert summary["quality_metrics"]["voxels_removed_by_table_gate"] == 0


def test_run_pipeline_v4_protects_hilar_bridge_and_cleans_far_deep_venous(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (7, 9, 9)
    bridge_voxel = (3, 1, 1)
    surface_voxel = (3, 1, 6)
    far_deep_voxel = (3, 4, 4)
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                bridge_voxel: 125,
                surface_voxel: 120,
                far_deep_voxel: 120,
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                bridge_voxel: 100,
                surface_voxel: 120,
                far_deep_voxel: 120,
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            for index in [bridge_voxel, surface_voxel, far_deep_voxel]:
                mask[index] = True
                confidence[index] = 0.5
                recovery[index] = True
        return (
            mask,
            confidence,
            {"candidate_voxels": int(recovery.sum()), "kept_voxels": int(recovery.sum()), "kept_components": int(recovery.any()), "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:6, 1:8, 1:8] = 5
        label_arr[3, 1, 0] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v4.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 2\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    confidence_min: 0.78\n"
        "  deep_liver_cleanup:\n"
        "    enabled: true\n"
        "    min_anchor_distance_mm: 2\n"
        "    confidence_min: 0.78\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[bridge_voxel] == 2
    assert fused[surface_voxel] == 0
    assert fused[far_deep_voxel] == 0
    assert summary["quality_metrics"]["hilar_protection_voxels"] > 0
    assert summary["quality_metrics"]["portal_relabel_bridge_voxels"] == 1
    assert summary["quality_metrics"]["final_surface_cleanup_voxels"] == 1
    assert summary["quality_metrics"]["deep_liver_cleanup_voxels"] == 1
    assert summary["quality_metrics"]["outside_body_voxels"] == 0


def test_run_pipeline_v5_protects_bridge_keeps_branch_and_cleans_isolated_blob(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (7, 11, 13)
    bridge_voxel = (3, 1, 1)
    surface_voxel = (3, 1, 9)
    branch_voxels = [(3, 5, x) for x in range(3, 9)]
    blob_voxels = [(3, y, x) for y in (7, 8) for x in (8, 9)]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                bridge_voxel: 125,
                surface_voxel: 120,
                **{index: 120 for index in branch_voxels},
                **{index: 120 for index in blob_voxels},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                bridge_voxel: 100,
                surface_voxel: 120,
                **{index: 120 for index in branch_voxels},
                **{index: 120 for index in blob_voxels},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            for index in [bridge_voxel, surface_voxel, *branch_voxels, *blob_voxels]:
                mask[index] = True
                confidence[index] = 0.5
                recovery[index] = True
        return (
            mask,
            confidence,
            {"candidate_voxels": int(recovery.sum()), "kept_voxels": int(recovery.sum()), "kept_components": int(recovery.any()), "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:6, 1:10, 1:12] = 5
        label_arr[3, 1, 0] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v5.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 2\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    confidence_min: 0.78\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: true\n"
        "    max_component_volume_mm3: 8\n"
        "    max_component_elongation: 2.0\n"
        "    confidence_min: 0.78\n"
        "    anchor_dilation_mm: 1\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[bridge_voxel] == 2
    assert fused[surface_voxel] == 0
    for index in branch_voxels:
        assert fused[index] == 3
    for index in blob_voxels:
        assert fused[index] == 0
    assert summary["quality_metrics"]["portal_relabel_bridge_voxels"] == 1
    assert summary["quality_metrics"]["final_surface_cleanup_voxels"] == 1
    assert summary["quality_metrics"]["deep_liver_cleanup_voxels"] == 0
    assert summary["quality_metrics"]["isolated_blob_cleanup_voxels"] == 4
    assert summary["quality_metrics"]["outside_body_voxels"] == 0


def test_run_pipeline_v6_cleans_apex_surface_residue_and_preserves_branches(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (8, 12, 14)
    bridge_voxel = (3, 1, 1)
    final_surface_voxel = (3, 1, 10)
    apex_blob = [(6, y, x) for y in (2, 3, 4) for x in (9, 10, 11)]
    morph_surface_blob = [(z, 1, x) for z in (2, 3, 4) for x in (5, 6, 7)]
    branch_voxels = [(3, 6, x) for x in range(4, 11)]
    isolated_blob = [(3, y, x) for y in (8, 9) for x in (9, 10)]
    all_voxels = [bridge_voxel, final_surface_voxel, *apex_blob, *morph_surface_blob, *branch_voxels, *isolated_blob]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                bridge_voxel: 125,
                **{index: 120 for index in [final_surface_voxel, *apex_blob, *morph_surface_blob, *branch_voxels, *isolated_blob]},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                bridge_voxel: 100,
                **{index: 120 for index in [final_surface_voxel, *apex_blob, *morph_surface_blob, *branch_voxels, *isolated_blob]},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            for index in all_voxels:
                mask[index] = True
                confidence[index] = 0.55
                recovery[index] = True
            confidence[final_surface_voxel] = 0.45
        return (
            mask,
            confidence,
            {"candidate_voxels": int(recovery.sum()), "kept_voxels": int(recovery.sum()), "kept_components": int(recovery.any()), "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:7, 1:11, 1:13] = 5
        label_arr[3, 1, 0] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v6.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 2\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    confidence_min: 0.50\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: true\n"
        "    max_component_volume_mm3: 8\n"
        "    max_component_elongation: 2.0\n"
        "    confidence_min: 0.80\n"
        "    anchor_dilation_mm: 1\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    apex_fraction: 0.20\n"
        "    confidence_min: 0.82\n"
        "    max_component_volume_mm3: 16\n"
        "    max_component_elongation: 2.2\n"
        "    anchor_dilation_mm: 1\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[bridge_voxel] == 2
    assert fused[final_surface_voxel] == 0
    for index in [*apex_blob, *morph_surface_blob, *isolated_blob]:
        assert fused[index] == 0
    for index in branch_voxels:
        assert fused[index] == 3
    assert summary["quality_metrics"]["portal_relabel_bridge_voxels"] == 1
    assert summary["quality_metrics"]["final_surface_cleanup_voxels"] == 1
    assert summary["quality_metrics"]["deep_liver_cleanup_voxels"] == 0
    assert summary["quality_metrics"]["isolated_blob_cleanup_voxels"] == 4
    assert summary["quality_metrics"]["apex_surface_cleanup_voxels"] == 18
    assert summary["quality_metrics"]["apex_surface_cleanup_components"] == 2
    assert summary["quality_metrics"]["outside_body_voxels"] == 0


def test_run_pipeline_v7_cleans_outer_peripheral_blob_and_protects_smv_portal(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (5, 16, 16)
    smv_bridge = [(2, 1, x) for x in range(1, 5)]
    outer_blob = [(2, y, x) for y in (13, 14) for x in (13, 14)]
    outer_portal_blob = [(2, y, x) for y in (13, 14) for x in (9, 10)]
    outer_branch = [(2, 4, x) for x in range(7, 14)]
    liver_branch = [(2, 8, x) for x in range(5, 11)]
    all_voxels = [*smv_bridge, *outer_blob, *outer_portal_blob, *outer_branch, *liver_branch]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                **{index: 125 for index in smv_bridge},
                **{index: 120 for index in [*outer_blob, *outer_portal_blob, *outer_branch, *liver_branch]},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                **{index: 100 for index in smv_bridge},
                **{index: 120 for index in [*outer_blob, *outer_portal_blob, *outer_branch, *liver_branch]},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            for index in [*smv_bridge, *outer_blob, *outer_branch, *liver_branch]:
                mask[index] = True
                confidence[index] = 0.55
                recovery[index] = index in liver_branch
        if phase_name == "portal":
            for index in outer_portal_blob:
                mask[index] = True
                confidence[index] = 0.55
        return (
            mask,
            confidence,
            {"candidate_voxels": int(recovery.sum()), "kept_voxels": int(recovery.sum()), "kept_components": int(recovery.any()), "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[:, 6:12, 4:12] = 5
        label_arr[2, 1, 0] = 64
        label_arr[2, 4, 7] = 63
        label_arr[2, 13, 7] = 63
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v7.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 2\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 4\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: true\n"
        "    max_component_volume_mm3: 8\n"
        "    max_component_linearity: 2.2\n"
        "    confidence_min: 0.82\n"
        "    anchor_dilation_mm: 1\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    for index in smv_bridge:
        assert fused[index] == 2
    for index in [*outer_blob, *outer_portal_blob]:
        assert fused[index] == 0
    for index in outer_branch:
        assert fused[index] == 3
    for index in liver_branch:
        assert fused[index] == 3
    assert summary["quality_metrics"]["smv_portal_protection_voxels"] > summary["quality_metrics"]["hilar_protection_voxels"]
    assert summary["quality_metrics"]["outer_peripheral_cleanup_voxels"] == 8
    assert summary["quality_metrics"]["outer_peripheral_cleanup_components"] == 2
    assert summary["quality_metrics"]["outer_peripheral_cleanup_by_label"]["venous"] == 4
    assert summary["quality_metrics"]["outer_peripheral_cleanup_by_label"]["portal"] == 4
    assert summary["quality_metrics"]["outside_body_voxels"] == 0


def test_run_pipeline_v7_smv_protection_does_not_shield_liver_surface_cleanup(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (5, 12, 12)
    smv_bridge = (2, 1, 1)
    liver_surface_residue = (2, 5, 4)
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                smv_bridge: 125,
                liver_surface_residue: 120,
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                smv_bridge: 100,
                liver_surface_residue: 120,
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "venous":
            for index in [smv_bridge, liver_surface_residue]:
                mask[index] = True
                confidence[index] = 0.45
            recovery[liver_surface_residue] = True
        return (
            mask,
            confidence,
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_arr[1:4, 5:10, 3:9] = 5
        label_arr[2, 1, 0] = 64
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v7_surface.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 8\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    confidence_min: 0.78\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: true\n"
        "    max_component_volume_mm3: 8\n"
        "    max_component_linearity: 2.2\n"
        "    confidence_min: 0.82\n"
        "    anchor_dilation_mm: 1\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert fused[smv_bridge] == 2
    assert fused[liver_surface_residue] == 0
    assert summary["quality_metrics"]["smv_portal_protection_voxels"] > summary["quality_metrics"]["hilar_protection_voxels"]
    assert summary["quality_metrics"]["portal_relabel_bridge_voxels"] == 1
    assert summary["quality_metrics"]["final_surface_cleanup_voxels"] == 1


def test_run_pipeline_v8_bridges_smv_portal_and_audits_post_anchor_blob(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (3, 22, 24)
    left_portal = [(1, y, x) for y in (10, 11) for x in (3, 4)]
    right_portal = [(1, y, x) for y in (10, 11) for x in (16, 17)]
    bridge_candidate = [(1, 10, x) for x in range(5, 16)]
    post_anchor_blob = [(1, y, x) for y in range(2, 6) for x in range(18, 22)]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {**{index: 140 for index in bridge_candidate}, **{index: 120 for index in left_portal + right_portal + post_anchor_blob}},
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {**{index: 120 for index in bridge_candidate}, **{index: 120 for index in left_portal + right_portal + post_anchor_blob}},
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in bridge_candidate:
                mask[index] = True
                confidence[index] = 0.55
        return (
            mask,
            confidence,
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[1, 8:14, 7:14] = label_ids["liver"]
        label_arr[1, 10:12, 3:5] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[1, 10:12, 16:18] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[1, 2:6, 18:22] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v8.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 18\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: true\n"
        "    max_gap_mm: 20\n"
        "    corridor_radius_mm: 1.1\n"
        "    endpoint_min_volume_mm3: 1\n"
        "    min_portal_minus_venous_hu: 10\n"
        "    fallback_centerline_enabled: false\n"
        "    bridge_confidence: 0.85\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: true\n"
        "    organ_envelope_dilation_mm: 1\n"
        "    core_anchor_protection_mm: 0\n"
        "    min_component_volume_mm3: 8\n"
        "    max_component_linearity: 3\n"
        "    confidence_max: 1.01\n"
        "    organ_envelope_masks:\n"
        "      - liver\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    for index in left_portal + right_portal + bridge_candidate:
        assert fused[index] == 2
    for index in post_anchor_blob:
        assert fused[index] == 0
    assert summary["quality_metrics"]["smv_portal_bridge_repair_pairs"] == 1
    assert summary["quality_metrics"]["smv_portal_bridge_repair_voxels"] == len(bridge_candidate)
    assert summary["quality_metrics"]["post_anchor_peripheral_cleanup_voxels"] == len(post_anchor_blob)
    assert summary["quality_metrics"]["post_anchor_peripheral_cleanup_components"] == 1


def test_run_pipeline_v9_tube_fills_bridge_and_cleans_liver_surface_sheet_without_label(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (3, 24, 28)
    left_portal = [(1, y, x) for y in (10, 11, 12) for x in (3, 4)]
    right_portal = [(1, y, x) for y in (10, 11, 12) for x in (18, 19)]
    sparse_bridge = [(1, 11, x) for x in range(5, 18)]
    sheet_blob = [(1, y, x) for y in range(2, 6) for x in range(20, 26)]
    branch = [(1, 16, x) for x in range(8, 18)]
    tube_side_voxels = [(1, y, x) for y in (10, 12) for x in range(5, 18)]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                **{index: 140 for index in sparse_bridge + tube_side_voxels},
                **{index: 120 for index in left_portal + right_portal + sheet_blob + branch},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                **{index: 120 for index in sparse_bridge + tube_side_voxels},
                **{index: 120 for index in left_portal + right_portal + sheet_blob + branch},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in [*sparse_bridge, *sheet_blob, *branch]:
                mask[index] = True
                confidence[index] = 0.55
        return (
            mask,
            confidence,
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[1, 2:22, 2:26] = label_ids["liver"]
        label_arr[1, 10:13, 3:5] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[1, 10:13, 18:20] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[1, 16, 8:18] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v9.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  include_totalseg_vessel_anchors_in_output: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 18\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: false\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: true\n"
        "    max_gap_mm: 20\n"
        "    corridor_radius_mm: 1.1\n"
        "    endpoint_min_volume_mm3: 1\n"
        "    min_portal_minus_venous_hu: 10\n"
        "    fallback_centerline_enabled: false\n"
        "    bridge_confidence: 0.85\n"
        "    morphological_tube_fill_enabled: true\n"
        "    tube_radius_mm: 1.1\n"
        "    closing_radius_mm: 1.1\n"
        "    min_evidence_fraction: 0.35\n"
        "    max_fill_to_evidence_ratio: 2.5\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: false\n"
        "  liver_surface_sheet_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    min_component_volume_mm3: 8\n"
        "    max_component_volume_mm3: 64\n"
        "    max_component_linearity: 4.5\n"
        "    min_surface_fraction: 0.55\n"
        "    confidence_max: 1.01\n"
        "    core_anchor_protection_mm: 1\n"
        "    bridge_protection_mm: 1\n"
        "    target_labels:\n"
        "      - arterial\n"
        "      - portal\n"
        "      - venous\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    for index in sparse_bridge + tube_side_voxels:
        assert fused[index] == 2
    for index in sheet_blob:
        assert fused[index] == 0
    for index in branch:
        assert fused[index] == 3
    assert summary["quality_metrics"]["smv_portal_bridge_repair_morph_fill_voxels"] > 0
    assert summary["quality_metrics"]["liver_surface_sheet_cleanup_voxels"] == len(sheet_blob)


def test_run_pipeline_v9_preserves_deep_liver_branch_while_cleaning_surface_sheet(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (5, 18, 20)
    left_portal = [(1, y, x) for y in (7, 8) for x in (3, 4)]
    right_portal = [(1, y, x) for y in (7, 8) for x in (14, 15)]
    surface_sheet = [(1, y, x) for y in (1, 2, 3) for x in range(12, 17)]
    deep_branch = [(2, 8, x) for x in range(9, 18)]
    images = {
        "uid.arterial": _image_with_overrides(100, {}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            100,
            {
                **{index: 120 for index in [*left_portal, *right_portal, *surface_sheet, *deep_branch]},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            100,
            {
                **{index: 120 for index in [*left_portal, *right_portal, *surface_sheet, *deep_branch]},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        if phase_name == "portal":
            for index in left_portal + right_portal:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in surface_sheet + deep_branch:
                mask[index] = True
                confidence[index] = 0.55
        return (
            mask,
            confidence,
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[1, 1:16, 1:19] = label_ids["liver"]
        label_arr[1, 7:9, 3:5] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[1, 7:9, 14:16] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v9.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  include_totalseg_vessel_anchors_in_output: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 18\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: false\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: false\n"
        "    max_gap_mm: 20\n"
        "    corridor_radius_mm: 1.1\n"
        "    endpoint_min_volume_mm3: 1\n"
        "    min_portal_minus_venous_hu: 10\n"
        "    fallback_centerline_enabled: false\n"
        "    bridge_confidence: 0.85\n"
        "    morphological_tube_fill_enabled: true\n"
        "    tube_radius_mm: 1.1\n"
        "    closing_radius_mm: 1.1\n"
        "    min_evidence_fraction: 0.35\n"
        "    max_fill_to_evidence_ratio: 2.5\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: false\n"
        "  liver_surface_sheet_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    min_component_volume_mm3: 8\n"
        "    max_component_volume_mm3: 64\n"
        "    max_component_linearity: 4.5\n"
        "    min_surface_fraction: 0.55\n"
        "    confidence_max: 1.01\n"
        "    core_anchor_protection_mm: 1\n"
        "    bridge_protection_mm: 1\n"
        "    target_labels:\n"
        "      - venous\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    for index in left_portal + right_portal:
        assert fused[index] == 2
    for index in surface_sheet:
        assert fused[index] == 0
    for index in deep_branch:
        assert fused[index] == 3
    assert summary["quality_metrics"]["liver_surface_sheet_cleanup_voxels"] == len(surface_sheet)


def test_run_pipeline_v11_preserves_bridge_and_cleans_apex_subsurface_sheet(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (20, 32, 32)
    left_portal = [(6, y, x) for y in (14, 15, 16) for x in (6, 7)]
    right_portal = [(6, y, x) for y in (14, 15, 16) for x in (24, 25)]
    sparse_bridge = [(6, 15, x) for x in range(8, 24)]
    subsurface_sheet = [(16, y, x) for y in range(10, 14) for x in range(12, 16)]
    apex_branch = [(15, 22, x) for x in range(8, 26)]
    liver_body = [(z, y, x) for z in range(2, 19) for y in range(4, 28) for x in range(4, 28)]
    images = {
        "uid.arterial": _image_with_overrides(-1000, {index: 100 for index in liver_body}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 140 for index in sparse_bridge},
                **{index: 120 for index in left_portal + right_portal + subsurface_sheet + apex_branch},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 120 for index in sparse_bridge + subsurface_sheet + apex_branch},
                **{index: 120 for index in left_portal + right_portal},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in [*sparse_bridge, *subsurface_sheet, *apex_branch]:
                mask[index] = True
                confidence[index] = 0.55
                recovery[index] = True
        return (
            mask,
            confidence,
            {
                "candidate_voxels": int(recovery.sum()),
                "kept_voxels": int(recovery.sum()),
                "kept_components": int(recovery.any()),
                "rejected_components": 0,
            },
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[2:19, 4:28, 4:28] = label_ids["liver"]
        label_arr[6, 14:17, 6:8] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[6, 14:17, 24:26] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    config_path = tmp_path / "v11.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  body_closing_mm: 0\n"
        "  body_dilation_mm: 0\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  include_totalseg_vessel_anchors_in_output: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 18\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: true\n"
        "    min_portal_minus_venous_hu: 30\n"
        "    protected_min_portal_minus_venous_hu: 20\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 0\n"
        "    apex_fraction: 0\n"
        "    confidence_min: 0.82\n"
        "    max_component_volume_mm3: 96\n"
        "    max_component_elongation: 2.4\n"
        "    anchor_dilation_mm: 4\n"
        "  apex_subsurface_cleanup:\n"
        "    enabled: true\n"
        "    apex_fraction: 0.12\n"
        "    subsurface_min_depth_mm: 3\n"
        "    subsurface_max_depth_mm: 8\n"
        "    confidence_min: 0.80\n"
        "    min_component_volume_mm3: 8\n"
        "    max_component_volume_mm3: 64\n"
        "    max_component_linearity: 4.5\n"
        "    min_surface_fraction: 0.35\n"
        "    anchor_dilation_mm: 0\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: false\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: false\n"
        "  liver_surface_sheet_cleanup:\n"
        "    enabled: true\n"
        "    surface_depth_mm: 1.1\n"
        "    min_component_volume_mm3: 8\n"
        "    max_component_volume_mm3: 64\n"
        "    max_component_linearity: 4.5\n"
        "    min_surface_fraction: 0.55\n"
        "    confidence_max: 1.01\n"
        "    core_anchor_protection_mm: 1\n"
        "    bridge_protection_mm: 1\n"
        "    target_labels:\n"
        "      - portal\n"
        "      - venous\n",
        encoding="utf-8",
    )
    captured: dict[str, np.ndarray] = {}

    def fake_sheet_cleanup(multilabel, confidence, **kwargs):
        captured["bridge_mask"] = kwargs["bridge_mask"].copy()
        return {
            "liver_surface_sheet_cleanup_voxels": 0,
            "liver_surface_sheet_cleanup_components": 0,
            "liver_surface_sheet_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
            "liver_surface_sheet_cleanup_protected_voxels": 0,
            "liver_surface_sheet_cleanup_candidate_voxels": 0,
        }

    monkeypatch.setattr(pipeline, "apply_liver_surface_sheet_cleanup", fake_sheet_cleanup)
    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    for index in sparse_bridge:
        assert fused[index] == 2
        assert captured["bridge_mask"][index]
    for index in subsurface_sheet:
        assert fused[index] == 0
    for index in apex_branch:
        assert fused[index] == 3
    assert summary["quality_metrics"]["portal_relabel_voxels"] == len(sparse_bridge)
    assert summary["quality_metrics"]["portal_relabel_bridge_voxels"] == len(sparse_bridge)
    assert summary["quality_metrics"]["apex_subsurface_cleanup_voxels"] == len(subsurface_sheet)
    assert summary["quality_metrics"]["apex_subsurface_cleanup_components"] == 1


def test_run_pipeline_v13_reconnects_trunk_before_apex_cleanup_and_keeps_smv_bridge(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (20, 32, 32)
    left_portal = [(6, y, x) for y in (14, 15, 16) for x in (6, 7)]
    right_portal = [(6, y, x) for y in (14, 15, 16) for x in (24, 25)]
    sparse_bridge = [(6, 15, x) for x in range(8, 24)]
    trunk_left = [(15, 22, x) for x in range(8, 12)]
    trunk_gap = [(15, 22, x) for x in range(12, 17)]
    trunk_right = [(15, 22, x) for x in range(17, 26)]
    apex_sheet = [(16, y, x) for y in range(10, 14) for x in range(12, 16)]
    liver_body = [(z, y, x) for z in range(2, 19) for y in range(4, 28) for x in range(4, 28)]
    images = {
        "uid.arterial": _image_with_overrides(-1000, {index: 100 for index in liver_body}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 140 for index in sparse_bridge},
                **{index: 120 for index in left_portal + right_portal + trunk_left + trunk_right + apex_sheet},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 120 for index in sparse_bridge + trunk_left + trunk_right + apex_sheet},
                **{index: 120 for index in left_portal + right_portal},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in [*trunk_left, *trunk_right, *apex_sheet]:
                mask[index] = True
                confidence[index] = 0.55
            for index in [*trunk_gap, *apex_sheet]:
                recovery[index] = True
        return (
            mask,
            confidence,
            {
                "candidate_voxels": int(recovery.sum()),
                "kept_voxels": int(recovery.sum()),
                "kept_components": int(recovery.any()),
                "rejected_components": 0,
            },
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[2:19, 4:28, 4:28] = label_ids["liver"]
        label_arr[6, 14:17, 6:8] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[6, 14:17, 24:26] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    bridge_called = False
    reconnect_called = False
    captured: dict[str, np.ndarray] = {}

    def fake_bridge(multilabel, confidence, **_kwargs):
        nonlocal bridge_called
        bridge_called = True
        for index in sparse_bridge:
            multilabel[index] = 2
            confidence[index] = 1.0
        return {
            "smv_portal_bridge_repair_voxels": len(sparse_bridge),
            "smv_portal_bridge_repair_pairs": 1,
            "smv_portal_bridge_repair_fallback_voxels": 0,
            "smv_portal_bridge_repair_max_gap_mm": 5.0,
            "smv_portal_bridge_repair_evidence_voxels": len(sparse_bridge),
            "smv_portal_bridge_repair_morph_fill_voxels": 0,
            "smv_portal_bridge_repair_rejected_pairs": 0,
            "smv_portal_bridge_repair_rejected_by_reason": {
                "insufficient_evidence": 0,
                "excessive_fill": 0,
                "not_connected": 0,
            },
        }

    def fake_reconnect(multilabel, confidence, **kwargs):
        nonlocal reconnect_called
        reconnect_called = True
        assert bridge_called is True
        trunk_seed_mask = kwargs["trunk_seed_mask"].copy()
        candidate_mask = kwargs["candidate_mask"].copy()
        assert trunk_seed_mask[6, 15, 8:24].any()
        for index in trunk_gap:
            assert candidate_mask[index]
            multilabel[index] = 3
            confidence[index] = 0.86
        return {
            "intrahepatic_trunk_reconnect_voxels": len(trunk_gap),
            "intrahepatic_trunk_reconnect_pairs": 1,
            "intrahepatic_trunk_reconnect_max_gap_mm": 5.0,
            "intrahepatic_trunk_reconnect_evidence_voxels": len(trunk_gap),
            "intrahepatic_trunk_reconnect_morph_fill_voxels": len(trunk_gap),
            "intrahepatic_trunk_reconnect_rejected_pairs": 0,
            "intrahepatic_trunk_reconnect_rejected_by_reason": {
                "insufficient_evidence": 0,
                "excessive_fill": 0,
                "not_connected": 0,
            },
        }

    def fake_apex_subsurface_cleanup(multilabel, confidence, **kwargs):
        assert reconnect_called is True
        protection_mask = kwargs["protection_mask"]
        captured["protection_mask"] = protection_mask.copy()
        for index in trunk_gap:
            assert protection_mask[index]
            assert multilabel[index] == 3
        for index in apex_sheet:
            multilabel[index] = 0
            confidence[index] = 0.0
        return {
            "apex_subsurface_cleanup_voxels": len(apex_sheet),
            "apex_subsurface_cleanup_components": 1,
            "apex_subsurface_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": len(apex_sheet)},
            "apex_subsurface_cleanup_by_region": {"apex": len(apex_sheet), "subsurface": len(apex_sheet)},
            "apex_subsurface_cleanup_candidate_voxels": len(apex_sheet),
            "apex_subsurface_cleanup_protected_voxels": len(trunk_gap),
        }

    monkeypatch.setattr(pipeline, "apply_smv_portal_bridge_repair", fake_bridge)
    monkeypatch.setattr(pipeline, "apply_intrahepatic_trunk_reconnect", fake_reconnect, raising=False)
    monkeypatch.setattr(pipeline, "apply_apex_subsurface_cleanup", fake_apex_subsurface_cleanup)

    config_path = tmp_path / "v13.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  body_closing_mm: 0\n"
        "  body_dilation_mm: 0\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  include_totalseg_vessel_anchors_in_output: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 18\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: false\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  apex_subsurface_cleanup:\n"
        "    enabled: true\n"
        "    apex_fraction: 0.20\n"
        "    subsurface_min_depth_mm: 2\n"
        "    subsurface_max_depth_mm: 8\n"
        "    confidence_min: 0.80\n"
        "    min_component_volume_mm3: 120\n"
        "    max_component_volume_mm3: 1800\n"
        "    max_component_linearity: 4.5\n"
        "    min_surface_fraction: 0.30\n"
        "    anchor_dilation_mm: 2\n"
        "    protection_source: protected_trunk\n"
        "  intrahepatic_trunk_reconnect:\n"
        "    enabled: true\n"
        "    target_labels:\n"
        "      - portal\n"
        "      - venous\n"
        "    max_gap_mm: 18\n"
        "    corridor_radius_mm: 2.5\n"
        "    tube_radius_mm: 2.5\n"
        "    closing_radius_mm: 1.2\n"
        "    min_component_volume_mm3: 300\n"
        "    min_evidence_fraction: 0.25\n"
        "    max_fill_to_evidence_ratio: 2.0\n"
        "    bridge_confidence: 0.86\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: true\n"
        "    max_gap_mm: 30\n"
        "    corridor_radius_mm: 5\n"
        "    endpoint_min_volume_mm3: 300\n"
        "    min_portal_minus_venous_hu: 10\n"
        "    fallback_centerline_enabled: false\n"
        "    bridge_confidence: 0.85\n"
        "    morphological_tube_fill_enabled: true\n"
        "    tube_radius_mm: 3.0\n"
        "    closing_radius_mm: 1.5\n"
        "    min_evidence_fraction: 0.35\n"
        "    max_fill_to_evidence_ratio: 1.75\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: false\n"
        "  liver_surface_sheet_cleanup:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    for index in sparse_bridge:
        assert fused[index] == 2
    for index in trunk_gap:
        assert fused[index] == 3
        assert captured["protection_mask"][index]
    for index in apex_sheet:
        assert fused[index] == 0
    assert bridge_called is True
    assert reconnect_called is True
    assert summary["quality_metrics"]["smv_portal_bridge_repair_voxels"] == len(sparse_bridge)
    assert summary["quality_metrics"]["smv_portal_bridge_repair_pairs"] == 1
    assert summary["quality_metrics"]["intrahepatic_trunk_reconnect_voxels"] == len(trunk_gap)
    assert summary["quality_metrics"]["intrahepatic_trunk_reconnect_pairs"] == 1
    assert summary["quality_metrics"]["apex_subsurface_cleanup_voxels"] == len(apex_sheet)
    assert summary["quality_metrics"]["apex_subsurface_cleanup_components"] == 1


def test_run_pipeline_v14_gap_fill_protects_repair_but_not_apex_sheet(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (20, 32, 32)
    left_portal = [(6, y, x) for y in (14, 15, 16) for x in (6, 7)]
    right_portal = [(6, y, x) for y in (14, 15, 16) for x in (24, 25)]
    sparse_bridge = [(6, 15, x) for x in range(8, 24)]
    trunk_left = [(15, 22, x) for x in range(8, 12)]
    trunk_gap = [(15, 22, x) for x in range(12, 17)]
    trunk_right = [(15, 22, x) for x in range(17, 26)]
    apex_sheet = [(16, y, x) for y in range(10, 14) for x in range(12, 16)]
    liver_body = [(z, y, x) for z in range(2, 19) for y in range(4, 28) for x in range(4, 28)]
    images = {
        "uid.arterial": _image_with_overrides(-1000, {index: 100 for index in liver_body}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 140 for index in sparse_bridge},
                **{index: 120 for index in left_portal + right_portal + trunk_left + trunk_right + apex_sheet},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 120 for index in sparse_bridge + trunk_left + trunk_right + apex_sheet},
                **{index: 120 for index in left_portal + right_portal},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in [*trunk_left, *trunk_right, *apex_sheet]:
                mask[index] = True
                confidence[index] = 0.55
            for index in [*trunk_gap, *apex_sheet]:
                recovery[index] = True
        return (
            mask,
            confidence,
            {
                "candidate_voxels": int(recovery.sum()),
                "kept_voxels": int(recovery.sum()),
                "kept_components": int(recovery.any()),
                "rejected_components": 0,
            },
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[2:19, 4:28, 4:28] = label_ids["liver"]
        label_arr[6, 14:17, 6:8] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[6, 14:17, 24:26] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    bridge_called = False
    reconnect_called = False
    gap_fill_called = False
    captured: dict[str, np.ndarray] = {}

    def fake_bridge(multilabel, confidence, **_kwargs):
        nonlocal bridge_called
        bridge_called = True
        for index in sparse_bridge:
            multilabel[index] = 2
            confidence[index] = 1.0
        return {
            "smv_portal_bridge_repair_voxels": len(sparse_bridge),
            "smv_portal_bridge_repair_pairs": 1,
            "smv_portal_bridge_repair_fallback_voxels": 0,
            "smv_portal_bridge_repair_max_gap_mm": 5.0,
            "smv_portal_bridge_repair_evidence_voxels": len(sparse_bridge),
            "smv_portal_bridge_repair_morph_fill_voxels": 0,
            "smv_portal_bridge_repair_rejected_pairs": 0,
            "smv_portal_bridge_repair_rejected_by_reason": {
                "insufficient_evidence": 0,
                "excessive_fill": 0,
                "not_connected": 0,
            },
        }

    def fake_reconnect(multilabel, confidence, **kwargs):
        nonlocal reconnect_called
        reconnect_called = True
        assert bridge_called is True
        assert kwargs["trunk_seed_mask"][6, 15, 8:24].any()
        return {
            "intrahepatic_trunk_reconnect_voxels": 0,
            "intrahepatic_trunk_reconnect_pairs": 0,
            "intrahepatic_trunk_reconnect_max_gap_mm": 0.0,
            "intrahepatic_trunk_reconnect_evidence_voxels": 0,
            "intrahepatic_trunk_reconnect_morph_fill_voxels": 0,
            "intrahepatic_trunk_reconnect_rejected_pairs": 0,
            "intrahepatic_trunk_reconnect_rejected_by_reason": {
                "insufficient_evidence": 0,
                "excessive_fill": 0,
                "not_connected": 0,
            },
        }

    def fake_gap_fill(multilabel, confidence, **kwargs):
        nonlocal gap_fill_called
        gap_fill_called = True
        assert reconnect_called is True
        assert bridge_called is True
        trunk_seed_mask = kwargs["trunk_seed_mask"].copy()
        candidate_mask = kwargs["candidate_mask"].copy()
        assert trunk_seed_mask[6, 15, 8:24].any()
        for index in trunk_gap:
            assert candidate_mask[index]
            multilabel[index] = 3
            confidence[index] = 0.86
        return {
            "intrahepatic_trunk_gap_fill_voxels": len(trunk_gap),
            "intrahepatic_trunk_gap_fill_components": 1,
            "intrahepatic_trunk_gap_fill_max_gap_mm": 5.0,
            "intrahepatic_trunk_gap_fill_rejected_components": 0,
        }

    def fake_apex_subsurface_cleanup(multilabel, confidence, **kwargs):
        assert gap_fill_called is True
        protection_mask = kwargs["protection_mask"]
        captured["protection_mask"] = protection_mask.copy()
        for index in trunk_gap:
            assert protection_mask[index]
            assert multilabel[index] == 3
        for index in apex_sheet:
            assert not protection_mask[index]
            multilabel[index] = 0
            confidence[index] = 0.0
        return {
            "apex_subsurface_cleanup_voxels": len(apex_sheet),
            "apex_subsurface_cleanup_components": 1,
            "apex_subsurface_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": len(apex_sheet)},
            "apex_subsurface_cleanup_by_region": {"apex": len(apex_sheet), "subsurface": len(apex_sheet)},
            "apex_subsurface_cleanup_candidate_voxels": len(apex_sheet),
            "apex_subsurface_cleanup_protected_voxels": len(trunk_gap),
        }

    monkeypatch.setattr(pipeline, "apply_smv_portal_bridge_repair", fake_bridge)
    monkeypatch.setattr(pipeline, "apply_intrahepatic_trunk_reconnect", fake_reconnect, raising=False)
    monkeypatch.setattr(pipeline, "apply_intrahepatic_trunk_gap_fill", fake_gap_fill, raising=False)
    monkeypatch.setattr(pipeline, "apply_apex_subsurface_cleanup", fake_apex_subsurface_cleanup)

    config_path = tmp_path / "v14.yaml"
    config_path.write_text(
        "vessel_extraction:\n"
        "  body_closing_mm: 0\n"
        "  body_dilation_mm: 0\n"
        "  intrahepatic_recovery:\n"
        "    surface_prune_enabled: false\n"
        "  include_totalseg_vessel_anchors_in_output: false\n"
        "  hilar_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  smv_portal_bridge_protection:\n"
        "    enabled: true\n"
        "    distance_mm: 1\n"
        "  portal_from_venous_relabel:\n"
        "    enabled: false\n"
        "  final_liver_surface_cleanup:\n"
        "    enabled: false\n"
        "  deep_liver_cleanup:\n"
        "    enabled: false\n"
        "  isolated_liver_blob_cleanup:\n"
        "    enabled: false\n"
        "  apex_surface_morph_cleanup:\n"
        "    enabled: false\n"
        "  apex_subsurface_cleanup:\n"
        "    enabled: true\n"
        "    apex_fraction: 0.70\n"
        "    subsurface_min_depth_mm: 2\n"
        "    subsurface_max_depth_mm: 18\n"
        "    confidence_min: 0.86\n"
        "    min_component_volume_mm3: 16\n"
        "    max_component_volume_mm3: 6000\n"
        "    max_component_linearity: 6.0\n"
        "    min_surface_fraction: 0.20\n"
        "    anchor_dilation_mm: 0\n"
        "    protection_source: bridge_and_reconnect\n"
        "  intrahepatic_trunk_reconnect:\n"
        "    enabled: true\n"
        "    target_labels:\n"
        "      - portal\n"
        "      - venous\n"
        "    max_gap_mm: 18\n"
        "    corridor_radius_mm: 2.5\n"
        "    tube_radius_mm: 2.5\n"
        "    closing_radius_mm: 1.2\n"
        "    min_component_volume_mm3: 300\n"
        "    min_evidence_fraction: 0.25\n"
        "    max_fill_to_evidence_ratio: 2.0\n"
        "    bridge_confidence: 0.86\n"
        "  intrahepatic_trunk_gap_fill:\n"
        "    enabled: true\n"
        "    target_labels:\n"
        "      - portal\n"
        "      - venous\n"
        "    max_gap_mm: 14\n"
        "    contact_radius_mm: 1.5\n"
        "    min_component_volume_mm3: 1\n"
        "    max_component_volume_mm3: 800\n"
        "    max_component_linearity: 8.0\n"
        "    min_contact_components: 2\n"
        "    bridge_confidence: 0.86\n"
        "  outer_peripheral_blob_cleanup:\n"
        "    enabled: false\n"
        "  smv_portal_bridge_repair:\n"
        "    enabled: true\n"
        "    max_gap_mm: 30\n"
        "    corridor_radius_mm: 5\n"
        "    endpoint_min_volume_mm3: 300\n"
        "    min_portal_minus_venous_hu: 10\n"
        "    fallback_centerline_enabled: false\n"
        "    bridge_confidence: 0.85\n"
        "    morphological_tube_fill_enabled: true\n"
        "    tube_radius_mm: 3.0\n"
        "    closing_radius_mm: 1.5\n"
        "    min_evidence_fraction: 0.35\n"
        "    max_fill_to_evidence_ratio: 1.75\n"
        "  post_anchor_peripheral_component_audit:\n"
        "    enabled: false\n"
        "  liver_surface_sheet_cleanup:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=config_path,
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    fused = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "compat_nifti" / "vessel_fused_multilabel.nii.gz")))

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    for index in sparse_bridge:
        assert fused[index] == 2
    for index in trunk_gap:
        assert fused[index] == 3
        assert captured["protection_mask"][index]
    for index in apex_sheet:
        assert fused[index] == 0
        assert not captured["protection_mask"][index]
    assert bridge_called is True
    assert reconnect_called is True
    assert gap_fill_called is True
    assert summary["quality_metrics"]["smv_portal_bridge_repair_voxels"] == len(sparse_bridge)
    assert summary["quality_metrics"]["smv_portal_bridge_repair_pairs"] == 1
    assert summary["quality_metrics"]["intrahepatic_trunk_connected_after"] is True
    assert summary["quality_metrics"]["intrahepatic_trunk_disconnected_components_after"] == 0
    assert summary["quality_metrics"]["intrahepatic_trunk_reconnect_voxels"] == 0
    assert summary["quality_metrics"]["intrahepatic_trunk_gap_fill_voxels"] == len(trunk_gap)
    assert summary["quality_metrics"]["intrahepatic_trunk_gap_fill_components"] == 1
    assert summary["quality_metrics"]["apex_subsurface_cleanup_voxels"] == len(apex_sheet)
    assert summary["quality_metrics"]["apex_subsurface_cleanup_components"] == 1


def test_run_pipeline_v10_does_not_enable_apex_subsurface_cleanup(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    shape_zyx = (20, 32, 32)
    left_portal = [(6, y, x) for y in (14, 15, 16) for x in (6, 7)]
    right_portal = [(6, y, x) for y in (14, 15, 16) for x in (24, 25)]
    sparse_bridge = [(6, 15, x) for x in range(8, 24)]
    subsurface_sheet = [(16, y, x) for y in range(10, 14) for x in range(12, 16)]
    apex_branch = [(15, 22, x) for x in range(8, 26)]
    liver_body = [(z, y, x) for z in range(2, 19) for y in range(4, 28) for x in range(4, 28)]
    images = {
        "uid.arterial": _image_with_overrides(-1000, {index: 100 for index in liver_body}, shape_zyx=shape_zyx),
        "uid.portal": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 140 for index in sparse_bridge},
                **{index: 120 for index in left_portal + right_portal + subsurface_sheet + apex_branch},
            },
            shape_zyx=shape_zyx,
        ),
        "uid.venous": _image_with_overrides(
            -1000,
            {
                **{index: 100 for index in liver_body},
                **{index: 120 for index in sparse_bridge + subsurface_sheet + apex_branch},
                **{index: 120 for index in left_portal + right_portal},
            },
            shape_zyx=shape_zyx,
        ),
    }
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        mask = np.zeros(shape, dtype=bool)
        confidence = np.zeros(shape, dtype=np.float32)
        recovery = np.zeros(shape, dtype=bool)
        if phase_name == "portal":
            for index in [*left_portal, *right_portal]:
                mask[index] = True
                confidence[index] = 0.9
        if phase_name == "venous":
            for index in [*sparse_bridge, *subsurface_sheet, *apex_branch]:
                mask[index] = True
                confidence[index] = 0.55
                recovery[index] = True
        return (
            mask,
            confidence,
            {
                "candidate_voxels": int(recovery.sum()),
                "kept_voxels": int(recovery.sum()),
                "kept_components": int(recovery.any()),
                "rejected_components": 0,
            },
            recovery,
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein"])
        label_arr[2:19, 4:28, 4:28] = label_ids["liver"]
        label_arr[6, 14:17, 6:8] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[6, 14:17, 24:26] = label_ids["portal_vein_and_splenic_vein"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)
    called = False

    def fake_apex_subsurface_cleanup(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("v10 baseline should not call apex_subsurface_cleanup")

    monkeypatch.setattr(pipeline, "apply_apex_subsurface_cleanup", fake_apex_subsurface_cleanup)

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=Path("config/ct0021_v10.yaml"),
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )

    assert summary["label"] is None
    assert summary["guidance_source"] == "auto_totalseg_priors"
    assert called is False
    assert summary["quality_metrics"]["apex_subsurface_cleanup_voxels"] == 0


def test_run_pipeline_v10_injects_totalseg_anchors_before_bridge_repair(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        SeriesInfo(
            series_uid="uid.arterial",
            n_slices=120,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="1",
        ),
        SeriesInfo(
            series_uid="uid.portal",
            n_slices=120,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="2",
        ),
        SeriesInfo(
            series_uid="uid.venous",
            n_slices=120,
            protocol_name="Venous Phase",
            series_description="1.0 x 1.0_A",
            convolution_kernel="B_SOFT_B",
            slice_thickness_mm=1.0,
            acquisition_time="3",
        ),
    ]
    images = {item.series_uid: _image(100) for item in candidates}
    monkeypatch.setattr(pipeline, "index_dicom_series", lambda _input: candidates)
    monkeypatch.setattr(pipeline, "_candidate_images_by_uid", lambda _candidates, _max_series: images)

    def fake_phase_candidate(image, reference, *, phase_name, **_kwargs):
        shape = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
        return (
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=np.float32),
            {"candidate_voxels": 0, "kept_voxels": 0, "kept_components": 0, "rejected_components": 0},
            np.zeros(shape, dtype=bool),
        )

    monkeypatch.setattr(pipeline, "_phase_candidate", fake_phase_candidate)

    def fake_totalseg(reference, output_dir, roi_subset, device, force=False):
        label_path = output_dir / "totalseg" / "roi_subset_multilabel.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_arr = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint16)
        label_ids = resolve_totalseg_label_ids(["liver", "portal_vein_and_splenic_vein", "inferior_vena_cava"])
        label_arr[:, 1:6, 1:6] = label_ids["liver"]
        label_arr[:, 2:4, 2:4] = label_ids["portal_vein_and_splenic_vein"]
        label_arr[:, 0, 0] = label_ids["inferior_vena_cava"]
        label = sitk.GetImageFromArray(label_arr)
        label.CopyInformation(reference)
        sitk.WriteImage(label, str(label_path))
        return label_path

    monkeypatch.setattr(pipeline, "ensure_totalseg_multilabel", fake_totalseg, raising=False)

    def fake_inject(multilabel, confidence, **_kwargs):
        multilabel[0, 0, 0] = 2
        confidence[0, 0, 0] = 1.0
        return {
            "totalseg_anchor_output_voxels": 1,
            "totalseg_anchor_output_by_phase": {
                "arterial": {"candidate_voxels": 0, "injected_voxels": 0},
                "portal": {"candidate_voxels": 1, "injected_voxels": 1},
                "venous": {"candidate_voxels": 0, "injected_voxels": 0},
            },
        }

    def fake_bridge(multilabel, confidence, **_kwargs):
        assert multilabel[0, 0, 0] == 2
        assert confidence[0, 0, 0] == 1.0
        return {
            "smv_portal_bridge_repair_voxels": 0,
            "smv_portal_bridge_repair_pairs": 0,
            "smv_portal_bridge_repair_fallback_voxels": 0,
            "smv_portal_bridge_repair_max_gap_mm": 0.0,
            "smv_portal_bridge_repair_evidence_voxels": 0,
            "smv_portal_bridge_repair_morph_fill_voxels": 0,
            "smv_portal_bridge_repair_rejected_pairs": 0,
            "smv_portal_bridge_repair_rejected_by_reason": {
                "insufficient_evidence": 0,
                "excessive_fill": 0,
                "not_connected": 0,
            },
        }

    monkeypatch.setattr(pipeline, "inject_totalseg_vessel_anchors", fake_inject)
    monkeypatch.setattr(pipeline, "apply_smv_portal_bridge_repair", fake_bridge)

    summary = pipeline.run_pipeline(
        input_path=tmp_path / "dicom",
        output_dir=tmp_path / "out",
        label_path=None,
        config_path=Path("config/ct0021_v10.yaml"),
        skip_frangi=True,
        skip_mesh=True,
        vesselness_mode=None,
        max_series=None,
    )
    assert summary["quality_metrics"]["totalseg_anchor_output_voxels"] == 1
