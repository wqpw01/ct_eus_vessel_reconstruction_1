from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SeriesInfo:
    series_uid: str
    n_slices: int
    protocol_name: str = ""
    series_description: str = ""
    convolution_kernel: str = ""
    slice_thickness_mm: float | None = None
    acquisition_time: str = ""
    series_time: str = ""
    body_part: str = ""
    frame_of_reference_uid: str = ""
    spacing_xy: tuple[float, float] | None = None
    z_range: tuple[float, float] | None = None
    direction: tuple[float, ...] | None = None
    files: tuple[Path, ...] = field(default_factory=tuple)


def _contains_any(text: str, keywords: list[str]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)


def filter_candidate_series(
    series: list[SeriesInfo],
    *,
    min_slices: int = 100,
    max_slice_thickness_mm: float = 1.5,
    soft_kernel_keywords: list[str] | None = None,
    excluded_protocol_keywords: list[str] | None = None,
) -> list[SeriesInfo]:
    soft_keywords = soft_kernel_keywords or ["SOFT", "STANDARD", "MED"]
    excluded_keywords = excluded_protocol_keywords or ["Scout", "Dose", "MIP", "lung"]
    candidates: list[SeriesInfo] = []
    for item in series:
        text = " ".join([item.protocol_name, item.series_description, item.convolution_kernel])
        if item.n_slices < min_slices:
            continue
        if item.slice_thickness_mm is None or item.slice_thickness_mm > max_slice_thickness_mm:
            continue
        if _contains_any(text, excluded_keywords):
            continue
        if item.convolution_kernel and not _contains_any(item.convolution_kernel, soft_keywords):
            continue
        candidates.append(item)
    return candidates


def _time_key(value: str) -> tuple[int, float | str]:
    if not value:
        return (1, "")
    try:
        return (0, float(value))
    except ValueError:
        return (0, value)


def sort_series_for_phase_analysis(series: list[SeriesInfo]) -> list[SeriesInfo]:
    return sorted(series, key=lambda item: (_time_key(item.acquisition_time), item.series_uid))
