import SimpleITK as sitk

from ct_eus_vessel.phase_scoring import score_phase_image


def _image_from_values(values: list[int]) -> sitk.Image:
    image = sitk.GetImageFromArray([[[values[0], values[1]], [values[2], values[3]]]])
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def test_score_phase_image_resamples_label_and_reports_roi_percentiles() -> None:
    image = _image_from_values([400, 300, 120, 200])
    label = _image_from_values([8, 25, 10, 15])

    scores = score_phase_image(
        series_uid="series-1",
        image=image,
        label_image=label,
        label_ids={"aorta": 8, "celiac_artery": 25, "portal_vein": 10, "liver_vein": 15, "ivc": 9},
        percentile=50,
        min_roi_voxels=1,
    )

    assert scores.series_uid == "series-1"
    assert scores.aorta == 400
    assert scores.celiac_artery == 300
    assert scores.portal_vein == 120
    assert scores.liver_vein == 200
    assert scores.ivc is None
