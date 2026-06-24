from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from ct_eus_vessel.series import SeriesInfo


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _sort_key(item: tuple[Path, pydicom.Dataset]) -> tuple[float, int, str]:
    path, ds = item
    z = None
    if hasattr(ds, "ImagePositionPatient"):
        position = _as_float_tuple(ds.ImagePositionPatient)
        if position and len(position) >= 3:
            z = position[2]
    instance = _as_float(getattr(ds, "InstanceNumber", None))
    return (
        z if z is not None else float("inf"),
        int(instance) if instance is not None else 10**9,
        path.name,
    )


def index_dicom_series(root: str | Path) -> list[SeriesInfo]:
    dicom_root = Path(root)
    grouped: dict[str, list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    for path in sorted(dicom_root.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not series_uid:
            continue
        grouped[series_uid].append((path, ds))

    indexed: list[SeriesInfo] = []
    for series_uid in sorted(grouped):
        items = sorted(grouped[series_uid], key=_sort_key)
        datasets = [ds for _, ds in items]
        first = datasets[0]
        z_values: list[float] = []
        for ds in datasets:
            position = _as_float_tuple(getattr(ds, "ImagePositionPatient", None))
            if position and len(position) >= 3:
                z_values.append(position[2])
        spacing_xy = _as_float_tuple(getattr(first, "PixelSpacing", None))
        direction = _as_float_tuple(getattr(first, "ImageOrientationPatient", None))
        indexed.append(
            SeriesInfo(
                series_uid=series_uid,
                n_slices=len(items),
                protocol_name=str(getattr(first, "ProtocolName", "") or ""),
                series_description=str(getattr(first, "SeriesDescription", "") or ""),
                convolution_kernel=str(getattr(first, "ConvolutionKernel", "") or ""),
                slice_thickness_mm=_as_float(getattr(first, "SliceThickness", None)),
                acquisition_time=str(getattr(first, "AcquisitionTime", "") or ""),
                series_time=str(getattr(first, "SeriesTime", "") or ""),
                body_part=str(getattr(first, "BodyPartExamined", "") or ""),
                frame_of_reference_uid=str(getattr(first, "FrameOfReferenceUID", "") or ""),
                spacing_xy=(spacing_xy[0], spacing_xy[1]) if spacing_xy and len(spacing_xy) >= 2 else None,
                z_range=(min(z_values), max(z_values)) if z_values else None,
                direction=direction,
                files=tuple(path for path, _ in items),
            )
        )
    return indexed
