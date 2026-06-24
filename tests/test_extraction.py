import numpy as np

from ct_eus_vessel.extraction import (
    compute_slice_frangi_vesselness,
    extract_vessel_candidate,
    fuse_phase_candidates,
    keep_components_near_anchors,
    keep_mask_within_anchor_distance,
    remove_small_components,
)


def test_remove_small_components_filters_by_physical_volume() -> None:
    mask = np.zeros((4, 8, 8), dtype=bool)
    mask[1, 1, 1] = True
    mask[1:3, 3:6, 3:6] = True

    cleaned = remove_small_components(mask, spacing_xyz=(1.0, 1.0, 1.0), min_volume_mm3=4)

    assert not cleaned[1, 1, 1]
    assert cleaned[1:3, 3:6, 3:6].all()


def test_extract_vessel_candidate_combines_hu_vesselness_and_exclusion_masks() -> None:
    image = np.zeros((3, 5, 5), dtype=np.float32)
    image[:, 2, 1:4] = 180
    image[:, 4, 4] = 900
    vesselness = np.zeros_like(image)
    vesselness[:, 2, 1:4] = 0.9
    hard_exclusion = np.zeros_like(image, dtype=bool)
    hard_exclusion[:, 2, 3] = True

    result = extract_vessel_candidate(
        image,
        vesselness=vesselness,
        spacing_xyz=(1.0, 1.0, 1.0),
        hu_low=80,
        hu_high=350,
        vesselness_min=0.5,
        hard_exclusion_mask=hard_exclusion,
        soft_penalty_mask=None,
        min_component_volume_mm3=1,
    )

    assert result.mask[:, 2, 1:3].all()
    assert not result.mask[:, 2, 3].any()
    assert not result.mask[:, 4, 4].any()
    assert result.confidence[:, 2, 1:3].max() > 0


def test_extract_vessel_candidate_hard_exclusion_overrides_high_vesselness() -> None:
    image = np.full((1, 3, 3), 180, dtype=np.float32)
    vesselness = np.ones_like(image)
    hard_exclusion = np.zeros_like(image, dtype=bool)
    hard_exclusion[0, 1, 1] = True

    result = extract_vessel_candidate(
        image,
        vesselness=vesselness,
        spacing_xyz=(1.0, 1.0, 1.0),
        hu_low=80,
        hu_high=350,
        vesselness_min=0.5,
        hard_exclusion_mask=hard_exclusion,
        soft_penalty_mask=None,
        min_component_volume_mm3=1,
    )

    assert not result.mask[0, 1, 1]
    assert result.confidence[0, 1, 1] == 0


def test_fuse_phase_candidates_preserves_labels_and_confidence() -> None:
    arterial = np.zeros((2, 3, 3), dtype=bool)
    portal = np.zeros_like(arterial)
    venous = np.zeros_like(arterial)
    arterial[:, 0, 0] = True
    portal[:, 1, 1] = True
    venous[:, 2, 2] = True

    fused = fuse_phase_candidates(
        arterial_mask=arterial,
        portal_mask=portal,
        venous_mask=venous,
        confidence_maps={
            "arterial": arterial.astype(float) * 0.7,
            "portal": portal.astype(float) * 0.8,
            "venous": venous.astype(float) * 0.9,
        },
    )

    assert set(np.unique(fused.multilabel)) == {0, 1, 2, 3}
    assert fused.multilabel[0, 0, 0] == 1
    assert fused.multilabel[0, 1, 1] == 2
    assert fused.multilabel[0, 2, 2] == 3
    assert fused.confidence[0, 2, 2] == 0.9


def test_compute_slice_frangi_vesselness_returns_normalized_volume() -> None:
    image = np.zeros((2, 16, 16), dtype=np.float32)
    image[:, 8, 3:13] = 250

    vesselness = compute_slice_frangi_vesselness(image, sigmas_voxels=[1.0, 2.0])

    assert vesselness.shape == image.shape
    assert vesselness.dtype == np.float32
    assert 0.0 <= float(vesselness.min()) <= float(vesselness.max()) <= 1.0


def test_keep_components_near_anchors_removes_distant_false_positive() -> None:
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 1:3, 1:3] = True
    mask[0, 5:7, 5:7] = True
    anchors = np.zeros_like(mask)
    anchors[0, 1, 1] = True

    kept = keep_components_near_anchors(mask, anchors, dilation_voxels=1)

    assert kept[0, 1:3, 1:3].all()
    assert not kept[0, 5:7, 5:7].any()


def test_keep_mask_within_anchor_distance_cuts_connected_bone_ring() -> None:
    mask = np.zeros((1, 9, 9), dtype=bool)
    mask[0, 4, 4] = True
    mask[0, 4, 5:9] = True
    anchors = np.zeros_like(mask)
    anchors[0, 4, 4] = True

    kept = keep_mask_within_anchor_distance(
        mask,
        anchors,
        spacing_xyz=(1.0, 1.0, 1.0),
        max_distance_mm=2.0,
    )

    assert kept[0, 4, 4]
    assert kept[0, 4, 6]
    assert not kept[0, 4, 7]
    assert not kept[0, 4, 8]
