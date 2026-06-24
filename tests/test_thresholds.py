import numpy as np

from ct_eus_vessel.phase import PhaseScores
from ct_eus_vessel.thresholds import anchor_hu_window, image_hu_window, phase_hu_window


def test_phase_hu_window_expands_for_bright_arterial_contrast() -> None:
    scores = PhaseScores(series_uid="arterial", aorta=492, celiac_artery=282, portal_vein=65)

    low, high = phase_hu_window(scores, phase="arterial", default_low=80, default_high=350)

    assert low == 140
    assert high == 620


def test_phase_hu_window_uses_portal_roi_for_portal_phase() -> None:
    scores = PhaseScores(series_uid="portal", aorta=183, portal_vein=210, liver_vein=145)

    low, high = phase_hu_window(scores, phase="portal", default_low=80, default_high=350)

    assert low == 105
    assert high == 350


def test_image_hu_window_uses_high_hu_proxy_outside_priors() -> None:
    image = np.array(
        [
            [[-1000, 40, 90, 120], [160, 200, 420, 900]],
            [[80, 110, 140, 180], [220, 260, 300, 340]],
        ],
        dtype=np.float32,
    )
    hard = np.zeros_like(image, dtype=bool)
    hard[0, 1, 3] = True
    soft = np.zeros_like(image, dtype=bool)
    soft[0, 1, 2] = True

    low, high = image_hu_window(
        image,
        default_low=80,
        default_high=350,
        hard_exclusion_mask=hard,
        soft_penalty_mask=soft,
        min_voxels=4,
    )

    assert 80 <= low <= 140
    assert high > 350
    assert high <= 700


def test_image_hu_window_falls_back_when_too_few_voxels() -> None:
    image = np.array([[[10, 20], [30, 90]]], dtype=np.float32)

    assert image_hu_window(
        image,
        default_low=80,
        default_high=350,
        hard_exclusion_mask=None,
        soft_penalty_mask=None,
        min_voxels=4,
    ) == (80, 350)


def test_anchor_hu_window_uses_vessel_anchor_not_bone_or_table() -> None:
    image = np.full((2, 6, 6), 520, dtype=np.float32)
    image[:, 2:4, 2:4] = 180
    anchors = np.zeros_like(image, dtype=bool)
    anchors[:, 2:4, 2:4] = True

    low, high = anchor_hu_window(
        image,
        anchor_mask=anchors,
        default_low=80,
        default_high=350,
        min_voxels=4,
    )

    assert low == 90
    assert high == 350
