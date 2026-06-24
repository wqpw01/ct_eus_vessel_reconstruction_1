import numpy as np

from ct_eus_vessel.intrahepatic import recover_intrahepatic_vessels


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "hu_low_margin": 20,
        "hu_high_cap": 350,
        "local_background_sigma_mm": 2.0,
        "local_contrast_min_hu": 8.0,
        "relaxed_vesselness_min": 0.005,
        "closing_mm": 0.0,
        "min_component_volume_mm3": 1.0,
        "anchor_dilation_mm": 1.0,
        "component_min_elongation": 2.0,
        "max_component_liver_fraction": 0.5,
    }


def test_recover_intrahepatic_vessels_keeps_low_frangi_tube_connected_to_anchor() -> None:
    image = np.full((1, 9, 9), 110, dtype=np.float32)
    image[0, 4, 2:8] = 165
    vesselness = np.zeros_like(image, dtype=np.float32)
    liver = np.ones_like(image, dtype=bool)
    body = np.ones_like(image, dtype=bool)
    hard = np.zeros_like(image, dtype=bool)
    anchors = np.zeros_like(image, dtype=bool)
    anchors[0, 4, 2] = True

    result = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=_config(),
    )

    assert result.mask[0, 4, 2:8].all()
    assert float(result.confidence[0, 4, 4]) > 0.35
    assert result.metrics["candidate_voxels"] >= 6


def test_recover_intrahepatic_vessels_rejects_blob_without_anchor_or_elongation() -> None:
    image = np.full((1, 9, 9), 110, dtype=np.float32)
    image[0, 4, 4] = 180
    vesselness = np.zeros_like(image, dtype=np.float32)
    liver = np.ones_like(image, dtype=bool)
    body = np.ones_like(image, dtype=bool)
    hard = np.zeros_like(image, dtype=bool)
    anchors = np.zeros_like(image, dtype=bool)

    result = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=_config(),
    )

    assert not result.mask.any()
    assert result.metrics["kept_components"] == 0


def test_recover_intrahepatic_vessels_never_keeps_hard_excluded_voxels() -> None:
    image = np.full((1, 9, 9), 110, dtype=np.float32)
    image[0, 4, 2:8] = 165
    vesselness = np.zeros_like(image, dtype=np.float32)
    liver = np.ones_like(image, dtype=bool)
    body = np.ones_like(image, dtype=bool)
    hard = np.zeros_like(image, dtype=bool)
    hard[0, 4, 5] = True
    anchors = np.zeros_like(image, dtype=bool)
    anchors[0, 4, 2] = True

    result = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=_config(),
    )

    assert not result.mask[0, 4, 5]
    assert result.confidence[0, 4, 5] == 0.0


def test_recover_intrahepatic_vessels_uses_phase_specific_contrast_threshold() -> None:
    image = np.full((1, 9, 9), 110, dtype=np.float32)
    image[0, 4, 2:8] = 140
    vesselness = np.zeros_like(image, dtype=np.float32)
    liver = np.ones_like(image, dtype=bool)
    body = np.ones_like(image, dtype=bool)
    hard = np.zeros_like(image, dtype=bool)
    anchors = np.zeros_like(image, dtype=bool)
    anchors[0, 4, 2] = True
    config = _config()
    config["phase_local_contrast_min_hu"] = {"portal": 50.0, "venous": 8.0}

    portal = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=config,
        phase_name="portal",
    )
    venous = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=config,
        phase_name="venous",
    )

    assert not portal.mask.any()
    assert venous.mask[0, 4, 2:8].all()


def test_recover_intrahepatic_vessels_filters_many_components_without_changing_rules() -> None:
    image = np.full((1, 18, 18), 110, dtype=np.float32)
    kept_branch = [(0, 3, x) for x in range(2, 9)]
    anchored_blob = [(0, y, x) for y in (10, 11) for x in (2, 3)]
    rejected_blob = [(0, y, x) for y in (12, 13) for x in (12, 13)]
    for index in kept_branch + anchored_blob + rejected_blob:
        image[index] = 180
    vesselness = np.zeros_like(image, dtype=np.float32)
    liver = np.ones_like(image, dtype=bool)
    body = np.ones_like(image, dtype=bool)
    hard = np.zeros_like(image, dtype=bool)
    anchors = np.zeros_like(image, dtype=bool)
    anchors[0, 10, 2] = True

    config = _config()
    config["anchor_dilation_mm"] = 1.0
    config["component_min_elongation"] = 3.0
    result = recover_intrahepatic_vessels(
        image,
        vesselness=vesselness,
        liver_mask=liver,
        body_mask=body,
        hard_exclusion_mask=hard,
        anchor_mask=anchors,
        hu_window=(120, 350),
        spacing_xyz=(1.0, 1.0, 1.0),
        config=config,
    )

    for index in kept_branch + anchored_blob:
        assert result.mask[index]
    for index in rejected_blob:
        assert not result.mask[index]
    assert result.metrics["kept_components"] == 2
    assert result.metrics["rejected_components"] >= 1
