import numpy as np

import ct_eus_vessel.postprocess as postprocess
from ct_eus_vessel.postprocess import (
    apply_liver_surface_recovery_gate,
    inject_totalseg_vessel_anchors,
)


def test_apply_liver_surface_recovery_gate_prunes_only_low_confidence_surface_recovery() -> None:
    multilabel = np.zeros((5, 7, 7), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:4, 1:6, 1:6] = True
    recovery = np.zeros(multilabel.shape, dtype=bool)

    low_surface = (1, 1, 1)
    high_surface = (1, 1, 2)
    low_deep = (2, 3, 3)
    non_recovery_surface = (1, 1, 3)
    for index in [low_surface, high_surface, low_deep, non_recovery_surface]:
        multilabel[index] = 3
    confidence[low_surface] = 0.50
    confidence[high_surface] = 0.90
    confidence[low_deep] = 0.50
    confidence[non_recovery_surface] = 0.50
    recovery[low_surface] = True
    recovery[high_surface] = True
    recovery[low_deep] = True

    metrics = apply_liver_surface_recovery_gate(
        multilabel,
        confidence,
        recovery_mask=recovery,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        confidence_min=0.75,
    )

    assert multilabel[low_surface] == 0
    assert confidence[low_surface] == 0.0
    assert multilabel[high_surface] == 3
    assert multilabel[low_deep] == 3
    assert multilabel[non_recovery_surface] == 3
    assert metrics["surface_pruned_voxels"] == 1
    assert metrics["surface_pruned_by_label"]["venous"] == 1


def test_inject_totalseg_vessel_anchors_adds_safe_extrahepatic_phase_labels() -> None:
    multilabel = np.zeros((1, 5, 5), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[0, 0, 0] = True
    hard[0, 4, 4] = True
    body[0, 0, 4] = False

    anchors = {
        "arterial": np.zeros(multilabel.shape, dtype=bool),
        "portal": np.zeros(multilabel.shape, dtype=bool),
        "venous": np.zeros(multilabel.shape, dtype=bool),
    }
    anchors["arterial"][0, 0, 0] = True
    anchors["portal"][0, 2, 2] = True
    anchors["portal"][0, 0, 4] = True
    anchors["venous"][0, 4, 4] = True
    multilabel[0, 2, 2] = 3

    metrics = inject_totalseg_vessel_anchors(
        multilabel,
        confidence,
        anchors_by_phase=anchors,
        body_mask=body,
        hard_exclusion_mask=hard,
        liver_mask=liver,
        enabled=True,
        include_liver=False,
    )

    assert multilabel[0, 2, 2] == 2
    assert confidence[0, 2, 2] == 1.0
    assert multilabel[0, 0, 0] == 0
    assert multilabel[0, 0, 4] == 0
    assert multilabel[0, 4, 4] == 0
    assert metrics["totalseg_anchor_output_voxels"] == 1
    assert metrics["totalseg_anchor_output_by_phase"]["portal"]["injected_voxels"] == 1
    assert metrics["totalseg_anchor_output_by_phase"]["arterial"]["candidate_voxels"] == 0


def test_final_liver_surface_cleanup_prunes_only_low_confidence_surface_venous() -> None:
    assert callable(getattr(postprocess, "apply_final_liver_surface_cleanup", None))
    cleanup = postprocess.apply_final_liver_surface_cleanup
    multilabel = np.zeros((5, 7, 7), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:4, 1:6, 1:6] = True

    low_surface = (1, 1, 1)
    high_surface = (1, 1, 2)
    low_deep = (2, 3, 3)
    portal_surface = (1, 1, 3)
    outside_liver = (0, 1, 1)
    for index in [low_surface, high_surface, low_deep, outside_liver]:
        multilabel[index] = 3
    multilabel[portal_surface] = 2
    confidence[low_surface] = 0.50
    confidence[high_surface] = 0.90
    confidence[low_deep] = 0.50
    confidence[portal_surface] = 0.50
    confidence[outside_liver] = 0.50

    metrics = cleanup(
        multilabel,
        confidence,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        confidence_min=0.78,
    )

    assert multilabel[low_surface] == 0
    assert confidence[low_surface] == 0.0
    assert multilabel[high_surface] == 3
    assert multilabel[low_deep] == 3
    assert multilabel[portal_surface] == 2
    assert multilabel[outside_liver] == 3
    assert metrics["final_liver_surface_cleanup_voxels"] == 1
    assert metrics["final_liver_surface_cleanup_by_label"]["venous"] == 1
    assert metrics["final_liver_surface_cleanup_by_label"]["portal"] == 0


def test_portal_from_venous_relabel_uses_liver_coverage_and_hu_margin() -> None:
    assert callable(getattr(postprocess, "apply_portal_from_venous_relabel", None))
    relabel = postprocess.apply_portal_from_venous_relabel
    multilabel = np.zeros((1, 3, 7), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.ones(multilabel.shape, dtype=bool)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 110.0, dtype=np.float32)

    relabel_voxel = (0, 1, 1)
    low_delta = (0, 1, 2)
    missing_portal = (0, 1, 3)
    missing_venous = (0, 1, 4)
    outside_liver = (0, 1, 5)
    already_portal = (0, 1, 6)
    for index in [relabel_voxel, low_delta, missing_portal, missing_venous, outside_liver]:
        multilabel[index] = 3
        confidence[index] = 0.6
    multilabel[already_portal] = 2
    confidence[already_portal] = 0.4
    portal_hu[relabel_voxel] = 170
    venous_hu[relabel_voxel] = 120
    portal_hu[missing_portal] = 180
    venous_hu[missing_portal] = 100
    portal_hu[missing_venous] = 180
    venous_hu[missing_venous] = 100
    portal_hu[outside_liver] = 180
    venous_hu[outside_liver] = 100
    portal_hu[already_portal] = 190
    venous_hu[already_portal] = 100
    portal_coverage[missing_portal] = False
    venous_coverage[missing_venous] = False
    liver[outside_liver] = False

    metrics = relabel(
        multilabel,
        confidence,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        liver_mask=liver,
        enabled=True,
        min_portal_minus_venous_hu=30.0,
    )

    assert multilabel[relabel_voxel] == 2
    assert confidence[relabel_voxel] == 0.6
    assert multilabel[low_delta] == 3
    assert multilabel[missing_portal] == 3
    assert multilabel[missing_venous] == 3
    assert multilabel[outside_liver] == 3
    assert multilabel[already_portal] == 2
    assert metrics["portal_relabel_voxels"] == 1
    assert metrics["portal_relabel_by_reason"]["eligible_venous_liver"] == 4
    assert metrics["portal_relabel_by_reason"]["missing_portal_coverage"] == 1
    assert metrics["portal_relabel_by_reason"]["missing_venous_coverage"] == 1
    assert metrics["portal_relabel_by_reason"]["insufficient_hu_margin"] == 1


def test_build_hilar_protection_mask_expands_around_anchors() -> None:
    assert callable(getattr(postprocess, "build_hilar_protection_mask", None))
    build_hilar_protection_mask = postprocess.build_hilar_protection_mask
    anchors = np.zeros((1, 1, 7), dtype=bool)
    anchors[0, 0, 1] = True

    protection = build_hilar_protection_mask(
        anchors,
        spacing_xyz=(1.0, 1.0, 1.0),
        protection_distance_mm=2.0,
    )

    assert protection[0, 0, 1]
    assert protection[0, 0, 2]
    assert protection[0, 0, 3]
    assert not protection[0, 0, 4]


def test_portal_from_venous_relabel_prefers_protected_bridge_threshold() -> None:
    relabel = postprocess.apply_portal_from_venous_relabel
    assert callable(getattr(postprocess, "build_hilar_protection_mask", None))
    build_hilar_protection_mask = postprocess.build_hilar_protection_mask
    multilabel = np.zeros((1, 1, 6), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.ones(multilabel.shape, dtype=bool)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 100.0, dtype=np.float32)
    anchors = np.zeros(multilabel.shape, dtype=bool)
    anchors[0, 0, 0] = True
    protection = build_hilar_protection_mask(
        anchors,
        spacing_xyz=(1.0, 1.0, 1.0),
        protection_distance_mm=2.0,
    )

    bridge = (0, 0, 1)
    far = (0, 0, 5)
    for index in [bridge, far]:
        multilabel[index] = 3
        confidence[index] = 0.6
        portal_hu[index] = 125.0
        venous_hu[index] = 100.0

    metrics = relabel(
        multilabel,
        confidence,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        liver_mask=liver,
        enabled=True,
        min_portal_minus_venous_hu=30.0,
        protection_mask=protection,
        protected_min_portal_minus_venous_hu=20.0,
    )

    assert multilabel[bridge] == 2
    assert multilabel[far] == 3
    assert metrics["portal_relabel_voxels"] == 1
    assert metrics["portal_relabel_bridge_voxels"] == 1


def test_final_liver_surface_cleanup_respects_hilar_protection_mask() -> None:
    cleanup = postprocess.apply_final_liver_surface_cleanup
    assert callable(getattr(postprocess, "build_hilar_protection_mask", None))
    build_hilar_protection_mask = postprocess.build_hilar_protection_mask
    multilabel = np.zeros((5, 7, 7), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:4, 1:6, 1:6] = True
    anchors = np.zeros(multilabel.shape, dtype=bool)
    anchors[1, 1, 1] = True
    protection = build_hilar_protection_mask(
        anchors,
        spacing_xyz=(1.0, 1.0, 1.0),
        protection_distance_mm=1.5,
    )

    protected_surface = (1, 1, 1)
    unprotected_surface = (1, 1, 4)
    deep_voxel = (2, 3, 3)
    for index in [protected_surface, unprotected_surface, deep_voxel]:
        multilabel[index] = 3
        confidence[index] = 0.5

    metrics = cleanup(
        multilabel,
        confidence,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        confidence_min=0.75,
        protection_mask=protection,
    )

    assert multilabel[protected_surface] == 3
    assert multilabel[unprotected_surface] == 0
    assert multilabel[deep_voxel] == 3
    assert metrics["final_liver_surface_cleanup_voxels"] == 1
    assert metrics["final_surface_cleanup_voxels"] == 1


def test_deep_liver_cleanup_prunes_low_confidence_far_venous() -> None:
    assert callable(getattr(postprocess, "apply_deep_liver_cleanup", None))
    cleanup = postprocess.apply_deep_liver_cleanup
    multilabel = np.zeros((1, 1, 8), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.ones(multilabel.shape, dtype=bool)
    anchors = np.zeros(multilabel.shape, dtype=bool)
    anchors[0, 0, 0] = True

    near = (0, 0, 1)
    far = (0, 0, 7)
    for index, value in [(near, 0.9), (far, 0.5)]:
        multilabel[index] = 3
        confidence[index] = value

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchors,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        min_anchor_distance_mm=2.0,
        confidence_min=0.75,
    )

    assert multilabel[near] == 3
    assert multilabel[far] == 0
    assert metrics["deep_liver_cleanup_voxels"] == 1


def test_isolated_liver_blob_cleanup_preserves_elongated_branch_and_protection() -> None:
    assert callable(getattr(postprocess, "apply_isolated_liver_blob_cleanup", None))
    cleanup = postprocess.apply_isolated_liver_blob_cleanup
    multilabel = np.zeros((1, 7, 9), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.ones(multilabel.shape, dtype=bool)
    anchors = np.zeros(multilabel.shape, dtype=bool)
    protection = np.zeros(multilabel.shape, dtype=bool)
    anchors[0, 3, 0] = True
    protection[0, 3, 1:3] = True

    branch = [(0, 1, x) for x in range(1, 7)]
    blob = [(0, y, x) for y in (5, 6) for x in (5, 6)]
    protected_blob = [(0, 3, 1), (0, 3, 2)]
    for index in branch + blob + protected_blob:
        multilabel[index] = 3
        confidence[index] = 0.5

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchors,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_component_volume_mm3=8.0,
        max_component_elongation=2.0,
        confidence_min=0.75,
        anchor_dilation_mm=1.0,
        protection_mask=protection,
    )

    for index in branch:
        assert multilabel[index] == 3
    for index in blob:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in protected_blob:
        assert multilabel[index] == 3
    assert metrics["isolated_blob_cleanup_voxels"] == 4
    assert metrics["isolated_blob_cleanup_components"] == 1
    assert metrics["isolated_blob_cleanup_by_label"]["venous"] == 4


def test_apex_surface_morph_cleanup_prunes_apex_and_surface_blobs_preserves_branches() -> None:
    assert callable(getattr(postprocess, "apply_apex_surface_morph_cleanup", None))
    cleanup = postprocess.apply_apex_surface_morph_cleanup
    multilabel = np.zeros((6, 12, 12), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:6, 1:11, 1:11] = True
    anchors = np.zeros(multilabel.shape, dtype=bool)
    protection = np.zeros(multilabel.shape, dtype=bool)
    anchors[3, 5, 1] = True
    protection[4, 2:4, 2:4] = True

    apex_blob = [(5, y, x) for y in (2, 3) for x in (8, 9)]
    surface_blob = [(z, 1, x) for z in (2, 3) for x in (8, 9)]
    branch = [(3, 6, x) for x in range(3, 10)]
    protected_blob = [(4, y, x) for y in (2, 3) for x in (2, 3)]
    anchor_branch = [(3, 5, x) for x in range(1, 5)]
    high_conf_surface = (3, 1, 5)
    portal_surface = (3, 1, 6)

    for index in [*apex_blob, *surface_blob, *branch, *protected_blob, *anchor_branch, high_conf_surface]:
        multilabel[index] = 3
        confidence[index] = 0.55
    multilabel[portal_surface] = 2
    confidence[portal_surface] = 0.55
    confidence[high_conf_surface] = 0.95

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchors,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        apex_fraction=0.25,
        confidence_min=0.82,
        max_component_volume_mm3=16.0,
        max_component_elongation=2.2,
        anchor_dilation_mm=1.0,
        protection_mask=protection,
    )

    for index in [*apex_blob, *surface_blob]:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in [*branch, *protected_blob, *anchor_branch]:
        assert multilabel[index] == 3
    assert multilabel[high_conf_surface] == 3
    assert multilabel[portal_surface] == 2
    assert metrics["apex_surface_cleanup_voxels"] == 8
    assert metrics["apex_surface_cleanup_components"] == 2
    assert metrics["apex_surface_cleanup_by_label"]["venous"] == 8
    assert metrics["apex_surface_cleanup_by_region"]["apex"] == 4
    assert metrics["apex_surface_cleanup_by_region"]["surface"] == 4


def test_build_smv_portal_protection_mask_expands_portal_anchor_inside_body() -> None:
    assert callable(getattr(postprocess, "build_smv_portal_protection_mask", None))
    build_protection = postprocess.build_smv_portal_protection_mask
    anchors = np.zeros((1, 1, 8), dtype=bool)
    body = np.ones(anchors.shape, dtype=bool)
    hard = np.zeros(anchors.shape, dtype=bool)
    anchors[0, 0, 1] = True
    body[0, 0, 4] = False
    hard[0, 0, 5] = True

    protection = build_protection(
        anchors,
        body_mask=body,
        hard_exclusion_mask=hard,
        spacing_xyz=(1.0, 1.0, 1.0),
        protection_distance_mm=3.0,
    )

    assert protection[0, 0, 1]
    assert protection[0, 0, 2]
    assert protection[0, 0, 3]
    assert not protection[0, 0, 4]
    assert not protection[0, 0, 5]
    assert not protection[0, 0, 6]


def test_outer_peripheral_blob_cleanup_prunes_only_unprotected_extrahepatic_blobs() -> None:
    assert callable(getattr(postprocess, "apply_outer_peripheral_blob_cleanup", None))
    cleanup = postprocess.apply_outer_peripheral_blob_cleanup
    multilabel = np.zeros((1, 12, 14), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[0, 2:8, 2:8] = True
    hard = np.zeros(multilabel.shape, dtype=bool)
    protection = np.zeros(multilabel.shape, dtype=bool)
    anchors = np.zeros(multilabel.shape, dtype=bool)

    peripheral_blob = [(0, y, x) for y in (9, 10) for x in (10, 11)]
    liver_blob = [(0, y, x) for y in (3, 4) for x in (3, 4)]
    protected_blob = [(0, y, x) for y in (9, 10) for x in (2, 3)]
    elongated_branch = [(0, 1, x) for x in range(2, 10)]
    hard_blob = [(0, y, x) for y in (1, 2) for x in (11, 12)]
    high_conf_blob = [(0, y, x) for y in (6, 7) for x in (10, 11)]
    body_outside_blob = [(0, 11, x) for x in (1, 2)]
    portal_blob = [(0, y, x) for y in (9, 10) for x in (6, 7)]

    for index in [
        *peripheral_blob,
        *liver_blob,
        *protected_blob,
        *elongated_branch,
        *hard_blob,
        *high_conf_blob,
        *body_outside_blob,
        *portal_blob,
    ]:
        multilabel[index] = 3
        confidence[index] = 0.5
    for index in portal_blob:
        multilabel[index] = 2
    for index in high_conf_blob:
        confidence[index] = 0.95
    for index in protected_blob:
        protection[index] = True
    for index in hard_blob:
        hard[index] = True
    for index in body_outside_blob:
        body[index] = False
    anchors[0, 1, 2] = True

    metrics = cleanup(
        multilabel,
        confidence,
        body_mask=body,
        liver_mask=liver,
        hard_exclusion_mask=hard,
        protection_mask=protection,
        anchor_mask=anchors,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_component_volume_mm3=8.0,
        max_component_linearity=2.2,
        confidence_min=0.80,
        anchor_dilation_mm=1.0,
    )

    for index in peripheral_blob:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in portal_blob:
        assert multilabel[index] == 0
    for index in liver_blob:
        assert multilabel[index] == 3
    for index in protected_blob:
        assert multilabel[index] == 3
    for index in elongated_branch:
        assert multilabel[index] == 3
    for index in hard_blob:
        assert multilabel[index] == 3
    for index in high_conf_blob:
        assert multilabel[index] == 3
    for index in body_outside_blob:
        assert multilabel[index] == 3
    assert metrics["outer_peripheral_cleanup_voxels"] == 8
    assert metrics["outer_peripheral_cleanup_components"] == 2
    assert metrics["outer_peripheral_cleanup_by_label"]["venous"] == 4
    assert metrics["outer_peripheral_cleanup_by_label"]["portal"] == 4


def test_smv_portal_bridge_repair_connects_large_portal_components_with_candidate_corridor() -> None:
    assert callable(getattr(postprocess, "apply_smv_portal_bridge_repair", None))
    repair = postprocess.apply_smv_portal_bridge_repair
    multilabel = np.zeros((1, 7, 14), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    protection = np.ones(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)

    left_component = [(0, y, x) for y in (2, 3) for x in (1, 2)]
    right_component = [(0, y, x) for y in (2, 3) for x in (10, 11)]
    corridor = [(0, 2, x) for x in range(3, 10)]
    for index in left_component + right_component:
        multilabel[index] = 2
        confidence[index] = 1.0
    for index in corridor:
        multilabel[index] = 3
        confidence[index] = 0.55
        portal_hu[index] = 140.0
        venous_hu[index] = 120.0

    metrics = repair(
        multilabel,
        confidence,
        body_mask=body,
        hard_exclusion_mask=hard,
        protection_mask=protection,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_gap_mm=10.0,
        corridor_radius_mm=1.1,
        endpoint_min_volume_mm3=1.0,
        min_portal_minus_venous_hu=10.0,
        fallback_centerline_enabled=False,
        bridge_confidence=0.85,
    )

    assert metrics["smv_portal_bridge_repair_pairs"] == 1
    assert metrics["smv_portal_bridge_repair_voxels"] == len(corridor)
    assert metrics["smv_portal_bridge_repair_fallback_voxels"] == 0
    for index in corridor:
        assert multilabel[index] == 2
        assert confidence[index] >= 0.85


def test_smv_portal_bridge_repair_tube_fill_requires_evidence_and_makes_thicker_bridge() -> None:
    repair = postprocess.apply_smv_portal_bridge_repair
    multilabel = np.zeros((1, 9, 16), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    protection = np.ones(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)

    left_component = [(0, y, x) for y in (3, 4, 5) for x in (1, 2)]
    right_component = [(0, y, x) for y in (3, 4, 5) for x in (13, 14)]
    sparse_evidence = [(0, 4, x) for x in range(3, 13)]
    for index in left_component + right_component:
        multilabel[index] = 2
        confidence[index] = 1.0
    for index in sparse_evidence:
        multilabel[index] = 3
        confidence[index] = 0.55
        portal_hu[index] = 140.0
        venous_hu[index] = 120.0
    for x in range(3, 13):
        for y in (3, 5):
            portal_hu[0, y, x] = 140.0
            venous_hu[0, y, x] = 120.0

    metrics = repair(
        multilabel,
        confidence,
        body_mask=body,
        hard_exclusion_mask=hard,
        protection_mask=protection,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_gap_mm=14.0,
        corridor_radius_mm=1.1,
        endpoint_min_volume_mm3=1.0,
        min_portal_minus_venous_hu=10.0,
        fallback_centerline_enabled=False,
        bridge_confidence=0.85,
        morphological_tube_fill_enabled=True,
        tube_radius_mm=1.1,
        closing_radius_mm=1.1,
        min_evidence_fraction=0.35,
        max_fill_to_evidence_ratio=2.0,
    )

    assert metrics["smv_portal_bridge_repair_pairs"] == 1
    assert metrics["smv_portal_bridge_repair_evidence_voxels"] >= len(sparse_evidence)
    assert metrics["smv_portal_bridge_repair_morph_fill_voxels"] > 0
    assert metrics["smv_portal_bridge_repair_voxels"] > len(sparse_evidence)
    for x in range(3, 13):
        assert multilabel[0, 4, x] == 2
        assert multilabel[0, 3, x] == 2
        assert multilabel[0, 5, x] == 2


def test_smv_portal_bridge_repair_rejects_tube_fill_without_evidence() -> None:
    repair = postprocess.apply_smv_portal_bridge_repair
    multilabel = np.zeros((1, 5, 12), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    protection = np.ones(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)
    for index in [(0, 2, 1), (0, 2, 2), (0, 2, 9), (0, 2, 10)]:
        multilabel[index] = 2
        confidence[index] = 1.0

    metrics = repair(
        multilabel,
        confidence,
        body_mask=body,
        hard_exclusion_mask=hard,
        protection_mask=protection,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_gap_mm=10.0,
        corridor_radius_mm=1.1,
        endpoint_min_volume_mm3=1.0,
        min_portal_minus_venous_hu=10.0,
        fallback_centerline_enabled=False,
        bridge_confidence=0.85,
        morphological_tube_fill_enabled=True,
        tube_radius_mm=1.1,
        closing_radius_mm=1.1,
        min_evidence_fraction=0.35,
        max_fill_to_evidence_ratio=1.75,
    )

    assert metrics["smv_portal_bridge_repair_pairs"] == 0
    assert metrics["smv_portal_bridge_repair_rejected_pairs"] == 1
    assert metrics["smv_portal_bridge_repair_rejected_by_reason"]["insufficient_evidence"] == 1
    assert metrics["smv_portal_bridge_repair_voxels"] == 0
    assert not multilabel[0, 2, 3:9].any()


def test_smv_portal_bridge_repair_uses_centerline_fallback_inside_protection_only() -> None:
    repair = postprocess.apply_smv_portal_bridge_repair
    multilabel = np.zeros((1, 5, 10), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    protection = np.zeros(multilabel.shape, dtype=bool)
    portal_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    venous_hu = np.full(multilabel.shape, 120.0, dtype=np.float32)
    portal_coverage = np.ones(multilabel.shape, dtype=bool)
    venous_coverage = np.ones(multilabel.shape, dtype=bool)

    left_component = [(0, 2, 1), (0, 2, 2)]
    right_component = [(0, 2, 7), (0, 2, 8)]
    protection[0, 1:4, 1:9] = True
    for index in left_component + right_component:
        multilabel[index] = 2
        confidence[index] = 1.0
    base_multilabel = multilabel.copy()
    base_confidence = confidence.copy()

    metrics = repair(
        multilabel,
        confidence,
        body_mask=body,
        hard_exclusion_mask=hard,
        protection_mask=protection,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_gap_mm=8.0,
        corridor_radius_mm=1.1,
        endpoint_min_volume_mm3=1.0,
        min_portal_minus_venous_hu=5.0,
        fallback_centerline_enabled=True,
        bridge_confidence=0.85,
    )

    assert metrics["smv_portal_bridge_repair_pairs"] == 1
    assert metrics["smv_portal_bridge_repair_fallback_voxels"] == 4
    for x in range(3, 7):
        assert multilabel[0, 2, x] == 2
    assert not multilabel[0, 0, 4]

    blocked = base_multilabel.copy()
    blocked_conf = base_confidence.copy()
    blocked_hard = hard.copy()
    blocked_hard[0, 2, 4] = True
    blocked_metrics = repair(
        blocked,
        blocked_conf,
        body_mask=body,
        hard_exclusion_mask=blocked_hard,
        protection_mask=protection,
        portal_hu=portal_hu,
        venous_hu=venous_hu,
        portal_coverage=portal_coverage,
        venous_coverage=venous_coverage,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        max_gap_mm=8.0,
        corridor_radius_mm=1.1,
        endpoint_min_volume_mm3=1.0,
        min_portal_minus_venous_hu=5.0,
        fallback_centerline_enabled=True,
        bridge_confidence=0.85,
    )

    assert blocked_metrics["smv_portal_bridge_repair_pairs"] == 0
    assert blocked[0, 2, 4] == 0


def test_liver_surface_sheet_cleanup_removes_sheet_and_preserves_branch_anchor_and_bridge() -> None:
    assert callable(getattr(postprocess, "apply_liver_surface_sheet_cleanup", None))
    cleanup = postprocess.apply_liver_surface_sheet_cleanup
    multilabel = np.zeros((3, 16, 16), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[:, 2:14, 2:14] = True
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    core_anchor = np.zeros(multilabel.shape, dtype=bool)
    bridge = np.zeros(multilabel.shape, dtype=bool)

    surface_sheet = [(1, y, x) for y in range(2, 6) for x in range(8, 14)]
    elongated_branch = [(1, 10, x) for x in range(4, 13)]
    anchor_sheet = [(1, y, x) for y in range(7, 10) for x in range(2, 5)]
    bridge_sheet = [(1, y, x) for y in range(11, 14) for x in range(9, 12)]
    deep_sheet = [(1, y, x) for y in range(7, 10) for x in range(7, 10)]
    for index in surface_sheet:
        multilabel[index] = 3
        confidence[index] = 1.0
    for index in elongated_branch:
        multilabel[index] = 3
        confidence[index] = 1.0
    for index in anchor_sheet:
        multilabel[index] = 2
        confidence[index] = 1.0
    for index in bridge_sheet:
        multilabel[index] = 2
        confidence[index] = 1.0
    for index in deep_sheet:
        multilabel[index] = 3
        confidence[index] = 1.0
    core_anchor[1, 8, 3] = True
    bridge[1, 12, 10] = True

    metrics = cleanup(
        multilabel,
        confidence,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        core_anchor_mask=core_anchor,
        bridge_mask=bridge,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        min_component_volume_mm3=8.0,
        max_component_volume_mm3=64.0,
        max_component_linearity=4.5,
        min_surface_fraction=0.55,
        confidence_max=1.01,
        core_anchor_protection_mm=2.0,
        bridge_protection_mm=2.0,
        target_labels=("arterial", "portal", "venous"),
    )

    for index in surface_sheet:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in elongated_branch + anchor_sheet + bridge_sheet + deep_sheet:
        assert multilabel[index] != 0
    assert metrics["liver_surface_sheet_cleanup_voxels"] == len(surface_sheet)
    assert metrics["liver_surface_sheet_cleanup_components"] == 1
    assert metrics["liver_surface_sheet_cleanup_by_label"]["venous"] == len(surface_sheet)
    assert metrics["liver_surface_sheet_cleanup_protected_voxels"] > 0
    assert metrics["liver_surface_sheet_cleanup_candidate_voxels"] > 0


def test_liver_surface_sheet_cleanup_prunes_surface_sheet_but_preserves_deep_branch() -> None:
    cleanup = postprocess.apply_liver_surface_sheet_cleanup
    multilabel = np.zeros((5, 14, 14), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:5, 1:13, 1:13] = True

    surface_sheet = [(1, y, x) for y in (1, 2, 3, 4) for x in range(7, 11)]
    deep_connector = [(2, 4, 7)]
    deep_branch = [(2, 5, x) for x in range(7, 11)]
    for index in [*surface_sheet, *deep_connector, *deep_branch]:
        multilabel[index] = 3
        confidence[index] = 0.5

    metrics = cleanup(
        multilabel,
        confidence,
        liver_mask=liver,
        body_mask=np.ones(multilabel.shape, dtype=bool),
        hard_exclusion_mask=np.zeros(multilabel.shape, dtype=bool),
        core_anchor_mask=np.zeros(multilabel.shape, dtype=bool),
        bridge_mask=np.zeros(multilabel.shape, dtype=bool),
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=1.1,
        min_component_volume_mm3=8.0,
        max_component_volume_mm3=64.0,
        max_component_linearity=4.5,
        min_surface_fraction=0.55,
        confidence_max=1.01,
        core_anchor_protection_mm=1.0,
        bridge_protection_mm=1.0,
        target_labels=("venous",),
    )

    for index in surface_sheet:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in [*deep_connector, *deep_branch]:
        assert multilabel[index] == 3
    assert metrics["liver_surface_sheet_cleanup_voxels"] == len(surface_sheet)
    assert metrics["liver_surface_sheet_cleanup_components"] == 1
    assert metrics["liver_surface_sheet_cleanup_by_label"]["venous"] == len(surface_sheet)


def test_apex_surface_morph_cleanup_preserves_protected_trunk_but_prunes_apex_sheet() -> None:
    cleanup = postprocess.apply_apex_surface_morph_cleanup
    multilabel = np.zeros((6, 16, 16), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:6, 1:15, 1:15] = True

    apex_sheet = [(4, y, x) for y in range(1, 5) for x in range(8, 13)]
    protected_trunk = [(2, 4, 9), (2, 5, 9), (2, 6, 9), (2, 7, 9), (2, 8, 9)]
    connected_sheet = [(2, 4, 9)]
    for index in [*apex_sheet, *protected_trunk, *connected_sheet]:
        multilabel[index] = 3
        confidence[index] = 0.5

    anchor_mask = np.zeros(multilabel.shape, dtype=bool)
    anchor_mask[2, 4, 9] = True

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchor_mask,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        surface_depth_mm=2.0,
        apex_fraction=0.25,
        confidence_min=0.8,
        max_component_volume_mm3=128.0,
        max_component_elongation=4.5,
        anchor_dilation_mm=1.0,
    )

    for index in apex_sheet:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in protected_trunk:
        assert multilabel[index] == 3
    assert metrics["apex_surface_cleanup_voxels"] == len(apex_sheet)
    assert metrics["apex_surface_cleanup_components"] == 1
    assert metrics["apex_surface_cleanup_by_label"]["venous"] == len(apex_sheet)


def test_apex_subsurface_cleanup_prunes_deeper_sheet_but_preserves_trunk() -> None:
    assert callable(getattr(postprocess, "apply_apex_subsurface_cleanup", None))
    cleanup = postprocess.apply_apex_subsurface_cleanup
    multilabel = np.zeros((6, 20, 20), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:6, 1:19, 1:19] = True

    trunk = [(2, y, 10) for y in range(4, 12)]
    sheet = [(3, y, x) for y in range(5, 9) for x in range(6, 10)]
    for index in [*trunk, *sheet]:
        multilabel[index] = 3
        confidence[index] = 0.5

    anchor_mask = np.zeros(multilabel.shape, dtype=bool)
    anchor_mask[2, 4, 10] = True

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchor_mask,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        apex_fraction=0.5,
        subsurface_min_depth_mm=1.0,
        subsurface_max_depth_mm=4.0,
        confidence_min=0.8,
        min_component_volume_mm3=4.0,
        max_component_volume_mm3=256.0,
        max_component_linearity=4.5,
        min_surface_fraction=0.35,
        anchor_dilation_mm=1.0,
    )

    for index in sheet:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in trunk:
        assert multilabel[index] == 3
    assert metrics["apex_subsurface_cleanup_voxels"] == len(sheet)
    assert metrics["apex_subsurface_cleanup_components"] == 1
    assert metrics["apex_subsurface_cleanup_by_label"]["venous"] == len(sheet)


def test_apex_subsurface_cleanup_expanded_window_removes_one_cm_sheet_but_preserves_protected_trunk() -> None:
    cleanup = postprocess.apply_apex_subsurface_cleanup
    multilabel = np.zeros((48, 48, 48), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[2:46, 2:46, 2:46] = True

    sheet = [(32, y, x) for y in range(10, 13) for x in range(18, 24)]
    protected_trunk = [(32, y, 30) for y in range(10, 13)]
    for index in [*sheet, *protected_trunk]:
        multilabel[index] = 3
        confidence[index] = 0.55

    protection_mask = np.zeros(multilabel.shape, dtype=bool)
    for index in protected_trunk:
        protection_mask[index] = True

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=np.zeros(multilabel.shape, dtype=bool),
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        apex_fraction=0.40,
        subsurface_min_depth_mm=8.0,
        subsurface_max_depth_mm=12.0,
        confidence_min=0.80,
        min_component_volume_mm3=8.0,
        max_component_volume_mm3=128.0,
        max_component_linearity=5.0,
        min_surface_fraction=0.30,
        anchor_dilation_mm=0.0,
        protection_mask=protection_mask,
    )

    for index in sheet:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in protected_trunk:
        assert multilabel[index] == 3
        assert confidence[index] == 0.55
    assert metrics["apex_subsurface_cleanup_voxels"] == len(sheet)
    assert metrics["apex_subsurface_cleanup_components"] == 1
    assert metrics["apex_subsurface_cleanup_by_region"]["subsurface"] == len(sheet)
    assert metrics["apex_subsurface_cleanup_protected_voxels"] >= len(protected_trunk)


def test_measure_intrahepatic_trunk_connectivity_reports_disconnected_large_branch() -> None:
    measure = postprocess.measure_intrahepatic_trunk_connectivity
    multilabel = np.zeros((5, 18, 24), dtype=np.uint8)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:5, 2:16, 2:22] = True
    trunk_seed = np.zeros(multilabel.shape, dtype=bool)

    trunk = [(2, 8, x) for x in range(3, 9)]
    branch = [(2, 8, x) for x in range(13, 20)]
    for index in trunk + branch:
        multilabel[index] = 3
    trunk_seed[2, 8, 4] = True

    metrics = measure(
        multilabel,
        trunk_seed_mask=trunk_seed,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        target_labels=("portal", "venous"),
        min_component_volume_mm3=4.0,
    )

    assert metrics["intrahepatic_trunk_connected"] is False
    assert metrics["intrahepatic_trunk_disconnected_components"] == 1
    assert metrics["intrahepatic_trunk_largest_disconnected_volume_mm3"] == 7.0
    assert metrics["intrahepatic_trunk_min_gap_mm"] == 5.0


def test_intrahepatic_trunk_reconnect_links_large_branch_with_evidence_corridor() -> None:
    repair = postprocess.apply_intrahepatic_trunk_reconnect
    multilabel = np.zeros((5, 20, 28), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:5, 2:18, 2:26] = True
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    trunk_seed = np.zeros(multilabel.shape, dtype=bool)

    trunk = [(2, 10, x) for x in range(4, 10)]
    branch = [(2, 10, x) for x in range(15, 22)]
    corridor = [(2, 10, x) for x in range(10, 15)]
    for index in trunk + branch:
        multilabel[index] = 3
        confidence[index] = 0.9
    trunk_seed[2, 10, 5] = True

    candidate = np.zeros(multilabel.shape, dtype=bool)
    for index in corridor:
        candidate[index] = True

    metrics = repair(
        multilabel,
        confidence,
        trunk_seed_mask=trunk_seed,
        candidate_mask=candidate,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        target_labels=("portal", "venous"),
        max_gap_mm=8.0,
        corridor_radius_mm=1.1,
        tube_radius_mm=1.1,
        closing_radius_mm=1.1,
        min_component_volume_mm3=4.0,
        min_evidence_fraction=0.20,
        max_fill_to_evidence_ratio=2.0,
        bridge_confidence=0.86,
    )

    for index in corridor:
        assert multilabel[index] == 3
        assert confidence[index] >= 0.86
    after = postprocess.measure_intrahepatic_trunk_connectivity(
        multilabel,
        trunk_seed_mask=trunk_seed,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        target_labels=("portal", "venous"),
        min_component_volume_mm3=4.0,
    )
    assert after["intrahepatic_trunk_connected"] is True
    assert metrics["intrahepatic_trunk_reconnect_pairs"] == 1
    assert metrics["intrahepatic_trunk_reconnect_voxels"] == len(corridor)


def test_intrahepatic_trunk_reconnect_rejects_without_evidence() -> None:
    repair = postprocess.apply_intrahepatic_trunk_reconnect
    multilabel = np.zeros((3, 14, 20), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.ones(multilabel.shape, dtype=bool)
    trunk_seed = np.zeros(multilabel.shape, dtype=bool)
    for index in [(1, 7, x) for x in range(2, 6)]:
        multilabel[index] = 3
    for index in [(1, 7, x) for x in range(12, 16)]:
        multilabel[index] = 3
    trunk_seed[1, 7, 3] = True

    metrics = repair(
        multilabel,
        confidence,
        trunk_seed_mask=trunk_seed,
        candidate_mask=np.zeros(multilabel.shape, dtype=bool),
        liver_mask=liver,
        body_mask=np.ones(multilabel.shape, dtype=bool),
        hard_exclusion_mask=np.zeros(multilabel.shape, dtype=bool),
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        target_labels=("venous",),
        max_gap_mm=10.0,
        corridor_radius_mm=1.1,
        tube_radius_mm=1.1,
        closing_radius_mm=1.1,
        min_component_volume_mm3=2.0,
        min_evidence_fraction=0.20,
        max_fill_to_evidence_ratio=2.0,
        bridge_confidence=0.86,
    )

    assert metrics["intrahepatic_trunk_reconnect_pairs"] == 0
    assert metrics["intrahepatic_trunk_reconnect_rejected_by_reason"]["insufficient_evidence"] == 1


def test_intrahepatic_trunk_gap_fill_repairs_internal_candidate_gap_in_connected_component() -> None:
    fill = postprocess.apply_intrahepatic_trunk_gap_fill
    multilabel = np.zeros((5, 22, 30), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[1:4, 2:20, 2:28] = True
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    trunk_seed = np.zeros(multilabel.shape, dtype=bool)

    left = [(2, 10, x) for x in range(4, 10)]
    right = [(2, 10, x) for x in range(15, 22)]
    detour = (
        [(2, y, 9) for y in range(11, 15)]
        + [(2, 14, x) for x in range(10, 16)]
        + [(2, y, 15) for y in range(11, 14)]
    )
    gap = [(2, 10, x) for x in range(10, 15)]
    for index in [*left, *right, *detour]:
        multilabel[index] = 3
        confidence[index] = 0.9
    trunk_seed[2, 10, 5] = True

    before = postprocess.measure_intrahepatic_trunk_connectivity(
        multilabel,
        trunk_seed_mask=trunk_seed,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        target_labels=("venous",),
        min_component_volume_mm3=4.0,
    )
    assert before["intrahepatic_trunk_connected"] is True

    candidate = np.zeros(multilabel.shape, dtype=bool)
    for index in gap:
        candidate[index] = True

    metrics = fill(
        multilabel,
        confidence,
        trunk_seed_mask=trunk_seed,
        candidate_mask=candidate,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        target_labels=("venous",),
        max_gap_mm=8.0,
        contact_radius_mm=1.1,
        min_component_volume_mm3=1.0,
        max_component_volume_mm3=16.0,
        max_component_linearity=8.0,
        min_contact_components=2,
        bridge_confidence=0.86,
    )

    for index in gap:
        assert multilabel[index] == 3
        assert confidence[index] >= 0.86
    assert metrics["intrahepatic_trunk_gap_fill_voxels"] == len(gap)
    assert metrics["intrahepatic_trunk_gap_fill_components"] == 1


def test_post_anchor_peripheral_component_audit_removes_large_outer_anchor_blob() -> None:
    assert callable(getattr(postprocess, "apply_post_anchor_peripheral_component_audit", None))
    audit = postprocess.apply_post_anchor_peripheral_component_audit
    multilabel = np.zeros((1, 16, 16), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    body = np.ones(multilabel.shape, dtype=bool)
    hard = np.zeros(multilabel.shape, dtype=bool)
    envelope_seed = np.zeros(multilabel.shape, dtype=bool)
    core_anchor = np.zeros(multilabel.shape, dtype=bool)

    envelope_seed[0, 2:5, 2:5] = True
    core_anchor[0, 3, 3] = True
    removable_blob = [(0, y, x) for y in range(11, 15) for x in range(11, 15)]
    protected_blob = [(0, y, x) for y in range(3, 5) for x in range(10, 12)]
    elongated_branch = [(0, 9, x) for x in range(4, 13)]
    hard_blob = [(0, y, x) for y in range(11, 13) for x in range(1, 3)]
    for index in removable_blob + protected_blob + elongated_branch + hard_blob:
        multilabel[index] = 2
        confidence[index] = 1.0
    for index in protected_blob:
        core_anchor[index] = True
    for index in hard_blob:
        hard[index] = True

    metrics = audit(
        multilabel,
        confidence,
        body_mask=body,
        hard_exclusion_mask=hard,
        envelope_seed_mask=envelope_seed,
        core_anchor_mask=core_anchor,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        organ_envelope_dilation_mm=2.0,
        core_anchor_protection_mm=1.0,
        min_component_volume_mm3=8.0,
        max_component_linearity=3.0,
        confidence_max=1.01,
    )

    for index in removable_blob:
        assert multilabel[index] == 0
        assert confidence[index] == 0.0
    for index in protected_blob + elongated_branch + hard_blob:
        assert multilabel[index] == 2
    assert metrics["post_anchor_peripheral_cleanup_voxels"] == len(removable_blob)
    assert metrics["post_anchor_peripheral_cleanup_components"] == 1
    assert metrics["post_anchor_peripheral_cleanup_by_label"]["portal"] == len(removable_blob)
