from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def read_dicom_series(files: tuple[Path, ...] | list[Path]) -> sitk.Image:
    if not files:
        raise ValueError("Cannot read an empty DICOM series")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    return reader.Execute()


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    *,
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024,
    pixel_id: int | None = None,
) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        moving.GetPixelID() if pixel_id is None else pixel_id,
    )


def array_to_like_image(
    array_zyx: np.ndarray,
    reference: sitk.Image,
    *,
    pixel_id: int,
) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx)
    image.CopyInformation(reference)
    return sitk.Cast(image, pixel_id)


def _image_physical_corners(image: sitk.Image) -> list[tuple[float, float, float]]:
    size = image.GetSize()
    corners: list[tuple[float, float, float]] = []
    for x in (0, size[0] - 1):
        for y in (0, size[1] - 1):
            for z in (0, size[2] - 1):
                corners.append(image.TransformIndexToPhysicalPoint((x, y, z)))
    return corners


def build_union_reference_image(
    images: list[sitk.Image],
    *,
    base_image: sitk.Image | None = None,
    pixel_id: int | None = None,
) -> sitk.Image:
    if not images:
        raise ValueError("Cannot build a union reference from no images")
    base = base_image if base_image is not None else images[0]
    continuous_indices = [
        base.TransformPhysicalPointToContinuousIndex(point)
        for image in images
        for point in _image_physical_corners(image)
    ]
    bounds = np.asarray(continuous_indices, dtype=float)
    min_index = np.floor(bounds.min(axis=0)).astype(int)
    max_index = np.ceil(bounds.max(axis=0)).astype(int)
    size = (max_index - min_index + 1).astype(int)

    reference = sitk.Image([int(v) for v in size], base.GetPixelID() if pixel_id is None else pixel_id)
    reference.SetSpacing(base.GetSpacing())
    reference.SetDirection(base.GetDirection())
    reference.SetOrigin(base.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in min_index)))
    return reference


def _coverage_mask(image: sitk.Image, reference: sitk.Image) -> np.ndarray:
    ones = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    ones.CopyInformation(image)
    ones += 1
    resampled = resample_to_reference(
        ones,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(resampled) > 0


def compose_reference_image(
    images: list[sitk.Image],
    reference: sitk.Image,
    *,
    default_value: float,
    pixel_id: int,
) -> sitk.Image:
    if not images:
        raise ValueError("Cannot compose a reference CT from no images")
    shape_zyx = (reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0])
    out = np.full(shape_zyx, default_value, dtype=np.float32)
    filled = np.zeros(shape_zyx, dtype=bool)
    for image in images:
        resampled = resample_to_reference(image, reference, interpolator=sitk.sitkLinear, default_value=default_value)
        valid = _coverage_mask(image, reference)
        take = valid & ~filled
        out[take] = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)[take]
        filled[take] = True
    return array_to_like_image(out, reference, pixel_id=pixel_id)


def image_to_slicer_ras_space(image: sitk.Image) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    oriented = orienter.Execute(image)

    ras_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(oriented))
    ras_image = sitk.Cast(ras_image, oriented.GetPixelID())
    ras_image.SetSpacing(oriented.GetSpacing())

    origin_lps = np.asarray(oriented.GetOrigin(), dtype=float)
    direction_lps = np.asarray(oriented.GetDirection(), dtype=float).reshape(3, 3)
    ras_image.SetOrigin(tuple(float(v) for v in _LPS_TO_RAS @ origin_lps))
    ras_image.SetDirection(tuple(float(v) for v in (_LPS_TO_RAS @ direction_lps).ravel()))
    ras_image.SetMetaData("NRRD_space", "right-anterior-superior")
    return ras_image
