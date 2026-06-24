from pathlib import Path

from ct_eus_vessel.config import load_config


def test_load_config_merges_default_and_override(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(
        "vessel_extraction:\n"
        "  hu_candidate_low: 95\n"
        "series_selection:\n"
        "  min_slices: 42\n",
        encoding="utf-8",
    )

    config = load_config(override)

    assert config["vessel_extraction"]["hu_candidate_low"] == 95
    assert config["vessel_extraction"]["hu_candidate_high"] == 350
    assert config["series_selection"]["min_slices"] == 42
    assert config["phase_detection"]["label_ids"]["aorta"] == 8


def test_default_config_includes_skeleton_exclusions_and_vessel_anchors() -> None:
    config = load_config()
    totalseg = config["totalsegmentator"]
    vessel = config["vessel_extraction"]
    recovery = vessel["intrahepatic_recovery"]
    hard = set(totalseg["hard_exclusion_masks"])
    roi_subset = set(totalseg["roi_subset"])

    for name in ["rib_left_1", "rib_right_12", "hip_left", "sacrum", "vertebrae_C1", "humerus_left"]:
        assert name in hard
        assert name in roi_subset
    assert "aorta" in totalseg["vessel_anchor_masks"]["arterial"]
    assert "portal_vein_and_splenic_vein" in totalseg["vessel_anchor_masks"]["portal"]
    assert "inferior_vena_cava" in totalseg["vessel_anchor_masks"]["venous"]
    assert vessel["include_totalseg_vessel_anchors_in_output"] is True
    assert vessel["totalseg_vessel_anchors_include_liver"] is False
    assert recovery["surface_prune_enabled"] is True
    assert recovery["surface_prune_depth_mm"] == 5
    assert recovery["surface_prune_confidence_min"] == 0.75


def test_ct0021_v3_config_enables_portal_relabel_and_final_surface_cleanup() -> None:
    config = load_config(Path("config/ct0021_v3.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["include_totalseg_vessel_anchors_in_output"] is True
    assert vessel["intrahepatic_recovery"]["surface_prune_depth_mm"] == 5
    assert vessel["portal_from_venous_relabel"] == {
        "enabled": True,
        "min_portal_minus_venous_hu": 30,
    }
    assert vessel["final_liver_surface_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 8,
        "confidence_min": 0.78,
    }


def test_ct0021_v4_config_enables_bridge_protection_and_deep_cleanup() -> None:
    config = load_config(Path("config/ct0021_v4.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["portal_from_venous_relabel"] == {
        "enabled": True,
        "min_portal_minus_venous_hu": 30,
        "protected_min_portal_minus_venous_hu": 20,
    }
    assert vessel["hilar_protection"] == {
        "enabled": True,
        "distance_mm": 12,
    }
    assert vessel["final_liver_surface_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 8,
        "confidence_min": 0.78,
    }
    assert vessel["deep_liver_cleanup"] == {
        "enabled": True,
        "min_anchor_distance_mm": 12,
        "confidence_min": 0.78,
    }


def test_ct0021_v5_config_keeps_bridge_protection_and_uses_blob_cleanup() -> None:
    config = load_config(Path("config/ct0021_v5.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["portal_from_venous_relabel"] == {
        "enabled": True,
        "min_portal_minus_venous_hu": 30,
        "protected_min_portal_minus_venous_hu": 20,
    }
    assert vessel["hilar_protection"] == {
        "enabled": True,
        "distance_mm": 12,
    }
    assert vessel["final_liver_surface_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 8,
        "confidence_min": 0.78,
    }
    assert vessel["deep_liver_cleanup"] == {
        "enabled": False,
    }
    assert vessel["isolated_liver_blob_cleanup"] == {
        "enabled": True,
        "max_component_volume_mm3": 32,
        "max_component_elongation": 2.0,
        "confidence_min": 0.78,
        "anchor_dilation_mm": 3,
    }


def test_ct0021_v6_config_adds_apex_surface_morph_cleanup() -> None:
    config = load_config(Path("config/ct0021_v6.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["portal_from_venous_relabel"] == {
        "enabled": True,
        "min_portal_minus_venous_hu": 30,
        "protected_min_portal_minus_venous_hu": 20,
    }
    assert vessel["hilar_protection"] == {
        "enabled": True,
        "distance_mm": 12,
    }
    assert vessel["deep_liver_cleanup"] == {
        "enabled": False,
    }
    assert vessel["isolated_liver_blob_cleanup"] == {
        "enabled": True,
        "max_component_volume_mm3": 48,
        "max_component_elongation": 2.0,
        "confidence_min": 0.80,
        "anchor_dilation_mm": 4,
    }
    assert vessel["apex_surface_morph_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 10,
        "apex_fraction": 0.12,
        "confidence_min": 0.82,
        "max_component_volume_mm3": 96,
        "max_component_elongation": 2.4,
        "anchor_dilation_mm": 4,
    }


def test_ct0021_v7_config_adds_outer_peripheral_cleanup_and_smv_protection() -> None:
    config = load_config(Path("config/ct0021_v7.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["hilar_protection"] == {
        "enabled": True,
        "distance_mm": 12,
    }
    assert vessel["smv_portal_bridge_protection"] == {
        "enabled": True,
        "distance_mm": 22,
    }
    assert vessel["deep_liver_cleanup"] == {
        "enabled": False,
    }
    assert vessel["outer_peripheral_blob_cleanup"] == {
        "enabled": True,
        "max_component_volume_mm3": 160,
        "max_component_linearity": 2.2,
        "confidence_min": 0.82,
        "anchor_dilation_mm": 6,
    }
    assert vessel["apex_surface_morph_cleanup"]["enabled"] is True


def test_ct0021_v8_config_adds_smv_bridge_repair_and_post_anchor_audit() -> None:
    config = load_config(Path("config/ct0021_v8.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["hilar_protection"] == {
        "enabled": True,
        "distance_mm": 12,
    }
    assert vessel["smv_portal_bridge_protection"] == {
        "enabled": True,
        "distance_mm": 22,
    }
    assert vessel["outer_peripheral_blob_cleanup"]["enabled"] is False
    assert vessel["smv_portal_bridge_repair"] == {
        "enabled": True,
        "max_gap_mm": 30,
        "corridor_radius_mm": 5,
        "endpoint_min_volume_mm3": 300,
        "min_portal_minus_venous_hu": 10,
        "fallback_centerline_enabled": True,
        "bridge_confidence": 0.85,
    }
    assert vessel["post_anchor_peripheral_component_audit"] == {
        "enabled": True,
        "organ_envelope_dilation_mm": 18,
        "core_anchor_protection_mm": 6,
        "min_component_volume_mm3": 96,
        "max_component_linearity": 3.0,
        "confidence_max": 1.01,
        "organ_envelope_masks": [
            "liver",
            "pancreas",
            "spleen",
            "stomach",
            "duodenum",
            "aorta",
            "inferior_vena_cava",
            "portal_vein_and_splenic_vein",
        ],
    }
    assert vessel["apex_surface_morph_cleanup"]["enabled"] is True


def test_ct0021_v9_config_adds_tube_fill_and_surface_sheet_cleanup() -> None:
    config = load_config(Path("config/ct0021_v9.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["smv_portal_bridge_repair"] == {
        "enabled": True,
        "max_gap_mm": 30,
        "corridor_radius_mm": 5,
        "endpoint_min_volume_mm3": 300,
        "min_portal_minus_venous_hu": 10,
        "fallback_centerline_enabled": False,
        "bridge_confidence": 0.85,
        "morphological_tube_fill_enabled": True,
        "tube_radius_mm": 3.0,
        "closing_radius_mm": 1.5,
        "min_evidence_fraction": 0.35,
        "max_fill_to_evidence_ratio": 1.75,
    }
    assert vessel["liver_surface_sheet_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 15,
        "min_component_volume_mm3": 200,
        "max_component_volume_mm3": 12000,
        "max_component_linearity": 4.5,
        "min_surface_fraction": 0.55,
        "confidence_max": 1.01,
        "core_anchor_protection_mm": 8,
        "bridge_protection_mm": 5,
        "target_labels": ["arterial", "portal", "venous"],
    }


def test_ct0021_v10_config_reverts_to_v9_baseline_without_apex_subsurface_cleanup() -> None:
    config = load_config(Path("config/ct0021_v10.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["smv_portal_bridge_repair"] == {
        "enabled": True,
        "max_gap_mm": 30,
        "corridor_radius_mm": 5,
        "endpoint_min_volume_mm3": 300,
        "min_portal_minus_venous_hu": 10,
        "fallback_centerline_enabled": False,
        "bridge_confidence": 0.85,
        "morphological_tube_fill_enabled": True,
        "tube_radius_mm": 3.0,
        "closing_radius_mm": 1.5,
        "min_evidence_fraction": 0.35,
        "max_fill_to_evidence_ratio": 1.75,
    }
    assert vessel["final_liver_surface_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 8,
        "confidence_min": 0.78,
    }
    assert vessel["apex_surface_morph_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 10,
        "apex_fraction": 0.12,
        "confidence_min": 0.82,
        "max_component_volume_mm3": 96,
        "max_component_elongation": 2.4,
        "anchor_dilation_mm": 4,
    }
    assert "apex_subsurface_cleanup" not in vessel
    assert vessel["liver_surface_sheet_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 15,
        "min_component_volume_mm3": 200,
        "max_component_volume_mm3": 12000,
        "max_component_linearity": 4.5,
        "min_surface_fraction": 0.55,
        "confidence_max": 1.01,
        "core_anchor_protection_mm": 8,
        "bridge_protection_mm": 5,
        "target_labels": ["arterial", "portal", "venous"],
    }


def test_ct0021_v11_config_adds_apex_subsurface_cleanup() -> None:
    config = load_config(Path("config/ct0021_v11.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["smv_portal_bridge_repair"]["enabled"] is True
    assert vessel["apex_surface_morph_cleanup"] == {
        "enabled": True,
        "surface_depth_mm": 4,
        "apex_fraction": 0.12,
        "confidence_min": 0.82,
        "max_component_volume_mm3": 96,
        "max_component_elongation": 2.4,
        "anchor_dilation_mm": 4,
    }
    assert vessel["apex_subsurface_cleanup"] == {
        "enabled": True,
        "apex_fraction": 0.12,
        "subsurface_min_depth_mm": 3,
        "subsurface_max_depth_mm": 8,
        "confidence_min": 0.80,
        "min_component_volume_mm3": 120,
        "max_component_volume_mm3": 1500,
        "max_component_linearity": 4.5,
        "min_surface_fraction": 0.35,
        "anchor_dilation_mm": 1,
    }
    assert vessel["liver_surface_sheet_cleanup"]["enabled"] is True
    assert vessel["liver_surface_sheet_cleanup"]["bridge_protection_mm"] == 12


def test_ct0021_v13_config_connects_trunk_and_expands_apex_subsurface_cleanup() -> None:
    config = load_config(Path("config/ct0021_v13.yaml"))
    vessel = config["vessel_extraction"]

    assert vessel["smv_portal_bridge_repair"]["enabled"] is True
    assert vessel["smv_portal_bridge_repair"]["morphological_tube_fill_enabled"] is True
    assert vessel["smv_portal_bridge_repair"]["fallback_centerline_enabled"] is False

    assert vessel["intrahepatic_trunk_reconnect"] == {
        "enabled": True,
        "target_labels": ["portal", "venous"],
        "max_gap_mm": 18,
        "corridor_radius_mm": 2.5,
        "tube_radius_mm": 2.5,
        "closing_radius_mm": 1.2,
        "min_component_volume_mm3": 300,
        "min_evidence_fraction": 0.25,
        "max_fill_to_evidence_ratio": 2.0,
        "bridge_confidence": 0.86,
    }

    assert vessel["apex_subsurface_cleanup"] == {
        "enabled": True,
        "apex_fraction": 0.20,
        "subsurface_min_depth_mm": 2,
        "subsurface_max_depth_mm": 8,
        "confidence_min": 0.80,
        "min_component_volume_mm3": 120,
        "max_component_volume_mm3": 1800,
        "max_component_linearity": 4.5,
        "min_surface_fraction": 0.30,
        "anchor_dilation_mm": 2,
        "protection_source": "protected_trunk",
    }
