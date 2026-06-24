from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    voxel_min_zyx: tuple[int, int, int]
    voxel_max_zyx: tuple[int, int, int]
    physical_min_xyz: tuple[float, float, float]
    physical_max_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_bbox(
    mask_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    padding_mm: float,
) -> BoundingBox | None:
    coords = np.argwhere(mask_zyx)
    if coords.size == 0:
        return None
    shape_zyx = np.array(mask_zyx.shape, dtype=int)
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    pad_vox_zyx = np.ceil(padding_mm / spacing_zyx).astype(int)
    min_zyx = np.maximum(coords.min(axis=0) - pad_vox_zyx, 0)
    max_zyx = np.minimum(coords.max(axis=0) + pad_vox_zyx + 1, shape_zyx)

    min_xyz = (
        origin_xyz[0] + float(min_zyx[2]) * spacing_xyz[0],
        origin_xyz[1] + float(min_zyx[1]) * spacing_xyz[1],
        origin_xyz[2] + float(min_zyx[0]) * spacing_xyz[2],
    )
    max_xyz = (
        origin_xyz[0] + float(max_zyx[2]) * spacing_xyz[0],
        origin_xyz[1] + float(max_zyx[1]) * spacing_xyz[1],
        origin_xyz[2] + float(max_zyx[0]) * spacing_xyz[2],
    )
    return BoundingBox(
        voxel_min_zyx=tuple(int(v) for v in min_zyx),
        voxel_max_zyx=tuple(int(v) for v in max_zyx),
        physical_min_xyz=tuple(round(float(v), 6) for v in min_xyz),
        physical_max_xyz=tuple(round(float(v), 6) for v in max_xyz),
    )
