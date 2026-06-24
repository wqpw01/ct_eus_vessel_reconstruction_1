import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.image_io import array_to_like_image, build_union_reference_image, compose_reference_image, image_to_slicer_ras_space, resample_to_reference


def test_array_to_like_image_preserves_reference_geometry() -> None:
    reference = sitk.Image([4, 5, 6], sitk.sitkInt16)
    reference.SetSpacing((0.5, 0.6, 1.5))
    reference.SetOrigin((1.0, 2.0, 3.0))
    reference.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    arr = np.ones((6, 5, 4), dtype=np.uint8)

    image = array_to_like_image(arr, reference, pixel_id=sitk.sitkUInt8)

    assert image.GetSize() == reference.GetSize()
    assert image.GetSpacing() == reference.GetSpacing()
    assert image.GetOrigin() == reference.GetOrigin()
    assert image.GetDirection() == reference.GetDirection()


def test_resample_to_reference_uses_reference_grid() -> None:
    moving = sitk.Image([2, 2, 2], sitk.sitkInt16)
    moving.SetSpacing((2.0, 2.0, 2.0))
    reference = sitk.Image([4, 4, 4], sitk.sitkInt16)
    reference.SetSpacing((1.0, 1.0, 1.0))

    out = resample_to_reference(moving, reference, interpolator=sitk.sitkLinear)

    assert out.GetSize() == (4, 4, 4)
    assert out.GetSpacing() == (1.0, 1.0, 1.0)


def test_image_to_slicer_ras_space_reorients_lps_image_to_ras_grid() -> None:
    arr = np.arange(2 * 4 * 3, dtype=np.uint8).reshape((2, 4, 3))
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((-10.0, -20.0, 30.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    ras_image = image_to_slicer_ras_space(image)

    assert ras_image.GetSize() == image.GetSize()
    assert ras_image.GetSpacing() == image.GetSpacing()
    assert ras_image.GetOrigin() == (6.0, 11.0, 30.0)
    assert ras_image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert ras_image.GetMetaData("NRRD_space") == "right-anterior-superior"
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ras_image), arr[:, ::-1, ::-1])


def test_union_reference_extends_to_superior_phase_coverage() -> None:
    portal = sitk.GetImageFromArray(np.full((3, 3, 4), 100, dtype=np.int16))
    portal.SetSpacing((1.0, 1.0, 1.0))
    portal.SetOrigin((0.0, 0.0, 0.0))
    portal.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    venous = sitk.GetImageFromArray(np.full((4, 3, 4), 200, dtype=np.int16))
    venous.SetSpacing((1.0, 1.0, 1.0))
    venous.SetOrigin((0.0, 0.0, 2.0))
    venous.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    reference = build_union_reference_image([portal, venous], base_image=portal)
    composite = compose_reference_image([portal, venous], reference, default_value=-1024, pixel_id=sitk.sitkInt16)

    assert reference.GetSize() == (4, 3, 6)
    assert reference.GetOrigin() == (0.0, 0.0, 0.0)
    composite_arr = sitk.GetArrayFromImage(composite)
    assert composite_arr.shape == (6, 3, 4)
    assert np.all(composite_arr[:3] == 100)
    assert np.all(composite_arr[3:] == 200)
