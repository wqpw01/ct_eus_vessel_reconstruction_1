import numpy as np

from ct_eus_vessel.geometry import compute_bbox


def test_compute_bbox_returns_voxel_and_physical_bounds_with_padding() -> None:
    mask = np.zeros((6, 8, 10), dtype=bool)
    mask[2:4, 3:6, 4:7] = True

    bbox = compute_bbox(
        mask,
        spacing_xyz=(0.5, 0.5, 2.0),
        origin_xyz=(10.0, 20.0, 30.0),
        padding_mm=1.0,
    )

    assert bbox.voxel_min_zyx == (1, 1, 2)
    assert bbox.voxel_max_zyx == (5, 8, 9)
    assert bbox.physical_min_xyz == (11.0, 20.5, 32.0)
    assert bbox.physical_max_xyz == (14.5, 24.0, 40.0)


def test_compute_bbox_returns_none_for_empty_mask() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    assert compute_bbox(mask, spacing_xyz=(1, 1, 1), origin_xyz=(0, 0, 0), padding_mm=2) is None
