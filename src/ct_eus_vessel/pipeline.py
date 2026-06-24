from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from ct_eus_vessel.config import load_config
from ct_eus_vessel.dicom_index import index_dicom_series
from ct_eus_vessel.extraction import compute_frangi_vesselness, compute_slice_frangi_vesselness, extract_vessel_candidate, fuse_phase_candidates, keep_components_near_anchors, keep_mask_within_anchor_distance, remove_small_components
from ct_eus_vessel.geometry import compute_bbox
from ct_eus_vessel.image_io import array_to_like_image, build_union_reference_image, compose_reference_image, image_to_slicer_ras_space, read_dicom_series, resample_to_reference
from ct_eus_vessel.intrahepatic import recover_intrahepatic_vessels
from ct_eus_vessel.masks import anchor_mask_from_weak_label, anchor_multilabel_from_weak_label, body_region_mask, bone_like_exclusion, masks_from_weak_label
from ct_eus_vessel.mesh import save_mask_mesh_ply
from ct_eus_vessel.phase import PhaseMapping, PhaseScores, choose_phase_series, choose_phase_series_from_metadata
from ct_eus_vessel.phase_scoring import score_phase_image
from ct_eus_vessel.postprocess import apply_apex_subsurface_cleanup, apply_apex_surface_morph_cleanup, apply_deep_liver_cleanup, apply_final_liver_surface_cleanup, apply_intrahepatic_trunk_reconnect, apply_isolated_liver_blob_cleanup, apply_liver_surface_recovery_gate, apply_liver_surface_sheet_cleanup, apply_outer_peripheral_blob_cleanup, apply_portal_from_venous_relabel, apply_post_anchor_peripheral_component_audit, apply_smv_portal_bridge_repair, build_hilar_protection_mask, build_smv_portal_protection_mask, inject_totalseg_vessel_anchors, measure_intrahepatic_trunk_connectivity
from ct_eus_vessel.qc import save_mip_png, save_overlay_png
from ct_eus_vessel.serialization import to_jsonable
from ct_eus_vessel.series import SeriesInfo, filter_candidate_series, sort_series_for_phase_analysis
from ct_eus_vessel.thresholds import anchor_hu_window, image_hu_window, phase_hu_window
from ct_eus_vessel.totalseg import TotalSegPriors, ensure_totalseg_multilabel, priors_from_totalseg_multilabel


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _spacing_xyz(image: sitk.Image) -> tuple[float, float, float]:
    spacing = image.GetSpacing()
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def _sigmas_mm_to_voxels(sigmas_mm: list[float], spacing_xyz: tuple[float, float, float]) -> list[float]:
    min_spacing = min(spacing_xyz)
    return [max(0.5, float(sigma_mm) / min_spacing) for sigma_mm in sigmas_mm]


def _candidate_images_by_uid(candidates: list[SeriesInfo], max_series: int | None) -> dict[str, sitk.Image]:
    selected = candidates[:max_series] if max_series is not None else candidates
    images: dict[str, sitk.Image] = {}
    for item in selected:
        images[item.series_uid] = read_dicom_series(item.files)
    return images


def _score_images(
    images: dict[str, sitk.Image],
    label_image: sitk.Image,
    config: dict[str, object],
) -> list[PhaseScores]:
    phase_cfg = config["phase_detection"]
    assert isinstance(phase_cfg, dict)
    return [
        score_phase_image(
            series_uid=series_uid,
            image=image,
            label_image=label_image,
            label_ids=phase_cfg["label_ids"],
            percentile=phase_cfg["score_percentile"],
            min_roi_voxels=phase_cfg["min_roi_voxels"],
        )
        for series_uid, image in images.items()
    ]


def _resampled_hu_and_coverage(image: sitk.Image, reference: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    resampled = resample_to_reference(image, reference, interpolator=sitk.sitkLinear, default_value=-1024)
    ones = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    ones.CopyInformation(image)
    ones += 1
    coverage_image = resample_to_reference(
        ones,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        pixel_id=sitk.sitkUInt8,
    )
    return (
        sitk.GetArrayFromImage(resampled).astype(np.float32),
        sitk.GetArrayFromImage(coverage_image).astype(bool),
    )


def _metadata_phase_scores(candidates: list[SeriesInfo], mapping: PhaseMapping) -> list[PhaseScores]:
    scores: list[PhaseScores] = []
    for item in candidates:
        scores.append(
            PhaseScores(
                series_uid=item.series_uid,
                aorta=100.0 if item.series_uid == mapping.arterial_uid else 0.0,
                celiac_artery=100.0 if item.series_uid == mapping.arterial_uid else 0.0,
                portal_vein=100.0 if item.series_uid == mapping.portal_uid else 0.0,
                ivc=100.0 if item.series_uid == mapping.venous_uid else 0.0,
                liver_vein=100.0 if item.series_uid == mapping.venous_uid else 0.0,
            )
        )
    return scores


def _label_masks(
    label_image: sitk.Image | None,
    reference: sitk.Image,
    config: dict[str, object],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if label_image is None:
        return None, None
    phase_cfg = config["phase_detection"]
    assert isinstance(phase_cfg, dict)
    ids = phase_cfg["label_ids"]
    vessel_ids = {ids["aorta"], ids["ivc"], ids["portal_vein"], ids["liver_vein"], ids["celiac_artery"]}
    organ_ids = {1, 2, 3, 4, 6, 7, 11, 14}
    return masks_from_weak_label(
        label_image,
        reference,
        organ_label_ids=organ_ids,
        vessel_label_ids=vessel_ids,
    )


def _totalseg_priors(
    reference: sitk.Image,
    *,
    output_dir: Path,
    config: dict[str, object],
    device: str | None,
    force: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, Path]:
    ts_cfg = config["totalsegmentator"]
    assert isinstance(ts_cfg, dict)
    ts_path = ensure_totalseg_multilabel(
        reference=reference,
        output_dir=output_dir,
        roi_subset=ts_cfg["roi_subset"],
        device=device or ts_cfg.get("device", "gpu"),
        force=force,
    )
    priors = priors_from_totalseg_multilabel(
        sitk.ReadImage(str(ts_path)),
        reference,
        soft_mask_names=ts_cfg["soft_penalty_masks"],
        hard_mask_names=ts_cfg["hard_exclusion_masks"],
        vessel_anchor_mask_names=ts_cfg.get("vessel_anchor_masks", {}),
    )
    return priors, ts_path


def _estimate_hu_windows_from_images(
    images: dict[str, sitk.Image],
    mapping: PhaseMapping,
    reference: sitk.Image,
    *,
    config: dict[str, object],
    hard_exclusion: np.ndarray | None,
    soft_penalty: np.ndarray | None,
    vessel_anchor_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, tuple[int, int]]:
    vessel_cfg = config["vessel_extraction"]
    assert isinstance(vessel_cfg, dict)
    out: dict[str, tuple[int, int]] = {}
    for phase_name, series_uid in [
        ("arterial", mapping.arterial_uid),
        ("portal", mapping.portal_uid),
        ("venous", mapping.venous_uid),
    ]:
        resampled = resample_to_reference(images[series_uid], reference, interpolator=sitk.sitkLinear)
        arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
        anchor_mask = (vessel_anchor_masks or {}).get(phase_name)
        if anchor_mask is not None and anchor_mask.any():
            out[phase_name] = anchor_hu_window(
                arr,
                anchor_mask=anchor_mask,
                default_low=vessel_cfg["hu_candidate_low"],
                default_high=vessel_cfg["hu_candidate_high"],
                min_voxels=vessel_cfg.get("anchor_hu_window_min_voxels", 64),
            )
        elif vessel_cfg.get("image_hu_window_without_anchor", False):
            out[phase_name] = image_hu_window(
                arr,
                default_low=vessel_cfg["hu_candidate_low"],
                default_high=vessel_cfg["hu_candidate_high"],
                hard_exclusion_mask=hard_exclusion,
                soft_penalty_mask=soft_penalty,
                min_voxels=vessel_cfg.get("hu_window_min_voxels", 512),
            )
        else:
            out[phase_name] = (int(vessel_cfg["hu_candidate_low"]), int(vessel_cfg["hu_candidate_high"]))
    return out


def _auto_anchor_mask(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    config: dict[str, object],
) -> np.ndarray:
    vessel_cfg = config["vessel_extraction"]
    assert isinstance(vessel_cfg, dict)
    anchors = (multilabel > 0) & (confidence >= vessel_cfg.get("auto_anchor_confidence_min", 0.35))
    if not anchors.any():
        return anchors
    return remove_small_components(
        anchors,
        spacing_xyz=spacing_xyz,
        min_volume_mm3=vessel_cfg.get("auto_anchor_min_component_volume_mm3", vessel_cfg["min_component_volume_mm3"]),
    )


def _phase_candidate(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    phase_name: str,
    config: dict[str, object],
    hu_window: tuple[int, int],
    vesselness_mode: str,
    anchor_mask: np.ndarray | None,
    hard_exclusion_mask: np.ndarray | None,
    soft_penalty_mask: np.ndarray | None,
    allowed_mask: np.ndarray | None = None,
    liver_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    resampled = resample_to_reference(image, reference, interpolator=sitk.sitkLinear)
    arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
    vessel_cfg = config["vessel_extraction"]
    assert isinstance(vessel_cfg, dict)
    hu_low, hu_high = hu_window
    spacing = _spacing_xyz(reference)
    if vesselness_mode == "hu":
        vesselness = np.clip(
            (arr - hu_low) / (hu_high - hu_low),
            0,
            1,
        )
    elif vesselness_mode == "slice-frangi":
        vesselness = compute_slice_frangi_vesselness(
            arr,
            sigmas_voxels=_sigmas_mm_to_voxels(vessel_cfg["frangi_sigmas_mm"], spacing),
        )
    elif vesselness_mode == "frangi3d":
        vesselness = compute_frangi_vesselness(
            arr,
            sigmas_voxels=_sigmas_mm_to_voxels(vessel_cfg["frangi_sigmas_mm"], spacing),
        )
    else:
        raise ValueError(f"Unsupported vesselness mode: {vesselness_mode}")
    bone_exclusion = bone_like_exclusion(
        arr,
        hu_threshold=vessel_cfg["hu_hard_high"],
        dilation_voxels=1,
        preserve_mask=anchor_mask,
    )
    combined_hard_exclusion = bone_exclusion if hard_exclusion_mask is None else (hard_exclusion_mask | bone_exclusion)
    if allowed_mask is not None:
        combined_hard_exclusion = combined_hard_exclusion | ~allowed_mask
    result = extract_vessel_candidate(
        arr,
        vesselness=vesselness,
        spacing_xyz=spacing,
        hu_low=hu_low,
        hu_high=hu_high,
        vesselness_min=0.15 if vesselness_mode == "hu" else 0.08,
        hard_exclusion_mask=combined_hard_exclusion,
        soft_penalty_mask=soft_penalty_mask,
        min_component_volume_mm3=vessel_cfg["min_component_volume_mm3"],
    )
    mask = result.mask
    confidence = result.confidence
    recovery_metrics = {
        "candidate_voxels": 0,
        "kept_voxels": 0,
        "kept_components": 0,
        "rejected_components": 0,
    }
    recovery_mask = np.zeros(arr.shape, dtype=bool)
    recovery_cfg = vessel_cfg.get("intrahepatic_recovery", {})
    if (
        isinstance(recovery_cfg, dict)
        and recovery_cfg.get("enabled", False)
        and phase_name in set(recovery_cfg.get("phases", []))
        and liver_mask is not None
    ):
        recovery = recover_intrahepatic_vessels(
            arr,
            vesselness=vesselness,
            liver_mask=liver_mask,
            body_mask=allowed_mask if allowed_mask is not None else np.ones(arr.shape, dtype=bool),
            hard_exclusion_mask=combined_hard_exclusion,
            anchor_mask=anchor_mask if anchor_mask is not None else np.zeros(arr.shape, dtype=bool),
            hu_window=hu_window,
            spacing_xyz=spacing,
            config=recovery_cfg,
            phase_name=phase_name,
        )
        recovery_mask = recovery.mask
        mask = mask | recovery.mask
        confidence = np.maximum(confidence, recovery.confidence)
        recovery_metrics = recovery.metrics
    return mask, confidence, recovery_metrics, recovery_mask


def _remove_table_like_components(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    enabled: bool,
) -> tuple[int, int]:
    if not enabled:
        return 0, 0
    mask = multilabel > 0
    if not mask.any():
        return 0, 0
    labeled, count = ndi.label(mask, structure=ndi.generate_binary_structure(mask.ndim, 1))
    z_size, y_size, x_size = mask.shape
    remove = np.zeros(mask.shape, dtype=bool)
    removed_components = 0
    for label_id in range(1, count + 1):
        coords = np.argwhere(labeled == label_id)
        if coords.size == 0:
            continue
        min_zyx = coords.min(axis=0)
        max_zyx = coords.max(axis=0) + 1
        extent = max_zyx - min_zyx
        touches_posterior = max_zyx[1] >= int(y_size * 0.88)
        spans_x = extent[2] >= int(x_size * 0.65)
        thin_y = extent[1] <= max(3, int(y_size * 0.18))
        spans_z = extent[0] >= int(z_size * 0.5)
        if touches_posterior and spans_x and thin_y and spans_z:
            remove |= labeled == label_id
            removed_components += 1
    removed_voxels = int(remove.sum())
    if removed_voxels:
        multilabel[remove] = 0
        confidence[remove] = 0.0
    return removed_components, removed_voxels


def _save_outputs(
    *,
    output_dir: Path,
    reference: sitk.Image,
    reference_arr: np.ndarray,
    arterial_mask: np.ndarray,
    portal_mask: np.ndarray,
    venous_mask: np.ndarray,
    multilabel: np.ndarray,
    confidence: np.ndarray,
    config: dict[str, object],
    skip_mesh: bool,
    quality_metrics: dict[str, int] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_dir = output_dir / "compat_nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(reference, str(nifti_dir / "reference_ct.nii.gz"))
    sitk.WriteImage(image_to_slicer_ras_space(reference), str(output_dir / "reference_ct.nrrd"))
    artery_image = array_to_like_image(arterial_mask.astype(np.uint8), reference, pixel_id=sitk.sitkUInt8)
    portal_image = array_to_like_image(portal_mask.astype(np.uint8), reference, pixel_id=sitk.sitkUInt8)
    venous_image = array_to_like_image(venous_mask.astype(np.uint8), reference, pixel_id=sitk.sitkUInt8)
    fused_image = array_to_like_image(multilabel.astype(np.uint8), reference, pixel_id=sitk.sitkUInt8)
    confidence_image = array_to_like_image(confidence.astype(np.float32), reference, pixel_id=sitk.sitkFloat32)
    sitk.WriteImage(artery_image, str(nifti_dir / "artery_candidate.nii.gz"))
    sitk.WriteImage(portal_image, str(nifti_dir / "portal_vein_candidate.nii.gz"))
    sitk.WriteImage(venous_image, str(nifti_dir / "systemic_vein_candidate.nii.gz"))
    sitk.WriteImage(fused_image, str(nifti_dir / "vessel_fused_multilabel.nii.gz"))
    sitk.WriteImage(confidence_image, str(nifti_dir / "vessel_confidence.nii.gz"))
    sitk.WriteImage(image_to_slicer_ras_space(artery_image), str(output_dir / "artery_candidate.nrrd"))
    sitk.WriteImage(image_to_slicer_ras_space(portal_image), str(output_dir / "portal_vein_candidate.nrrd"))
    sitk.WriteImage(image_to_slicer_ras_space(venous_image), str(output_dir / "systemic_vein_candidate.nrrd"))
    sitk.WriteImage(image_to_slicer_ras_space(fused_image), str(output_dir / "vessel_fused_multilabel.nrrd"))
    sitk.WriteImage(image_to_slicer_ras_space(confidence_image), str(output_dir / "vessel_confidence.nrrd"))

    fused_mask = multilabel > 0
    vessel_cfg = config["vessel_extraction"]
    bbox = compute_bbox(
        fused_mask,
        spacing_xyz=_spacing_xyz(reference),
        origin_xyz=tuple(float(v) for v in reference.GetOrigin()),
        padding_mm=vessel_cfg["bbox_padding_mm"],
    )
    bbox_payload = {"bbox": bbox.to_dict() if bbox is not None else None}
    _write_json(output_dir / "bbox.json", bbox_payload)

    save_overlay_png(ct_zyx=reference_arr, mask_zyx=fused_mask, output_path=output_dir / "qc" / "overlay_axial.png")
    save_mip_png(fused_mask, output_path=output_dir / "qc" / "mask_mip_axial.png")
    mesh_written = False if skip_mesh else save_mask_mesh_ply(
        fused_mask,
        reference=reference,
        output_path=output_dir / "mesh" / "vessel_fused_slicer_ras.ply",
        coordinate_system="slicer_ras",
    )

    return {
        "bbox": bbox_payload["bbox"],
        "mesh_written": mesh_written,
        "voxel_counts": {
            "arterial": int(arterial_mask.sum()),
            "portal": int(portal_mask.sum()),
            "venous": int(venous_mask.sum()),
            "fused": int(fused_mask.sum()),
        },
        "quality_metrics": quality_metrics or {},
    }


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    label_path: Path | None,
    config_path: Path | None,
    skip_frangi: bool,
    skip_mesh: bool = False,
    vesselness_mode: str | None = None,
    max_series: int | None,
    force_totalseg: bool = False,
    totalseg_device: str | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    if vesselness_mode is None:
        vesselness_mode = "hu" if skip_frangi else "slice-frangi"
    series = index_dicom_series(input_path)
    selection = config["series_selection"]
    candidates = sort_series_for_phase_analysis(
        filter_candidate_series(
            series,
            min_slices=selection["min_slices"],
            max_slice_thickness_mm=selection["max_slice_thickness_mm"],
            soft_kernel_keywords=selection["soft_kernel_keywords"],
            excluded_protocol_keywords=selection["excluded_protocol_keywords"],
        )
    )
    selected_candidates = candidates[:max_series] if max_series is not None else candidates
    if len(selected_candidates) < 3:
        raise RuntimeError(f"Need at least 3 candidate contrast series, found {len(selected_candidates)}")

    images = _candidate_images_by_uid(candidates, max_series)
    label_found = label_path
    label_image = sitk.ReadImage(str(label_found)) if label_found is not None else None
    manual_label = label_image is not None
    if manual_label:
        scores = _score_images(images, label_image, config)
        mapping = choose_phase_series(scores)
    else:
        mapping = choose_phase_series_from_metadata(selected_candidates)
        scores = _metadata_phase_scores(selected_candidates, mapping)
    scores_by_uid = {item.series_uid: item for item in scores}
    vessel_cfg = config["vessel_extraction"]
    hu_windows: dict[str, tuple[int, int]]
    if manual_label:
        hu_windows = {
            "arterial": phase_hu_window(scores_by_uid[mapping.arterial_uid], phase="arterial", default_low=vessel_cfg["hu_candidate_low"], default_high=vessel_cfg["hu_candidate_high"]),
            "portal": phase_hu_window(scores_by_uid[mapping.portal_uid], phase="portal", default_low=vessel_cfg["hu_candidate_low"], default_high=vessel_cfg["hu_candidate_high"]),
            "venous": phase_hu_window(scores_by_uid[mapping.venous_uid], phase="venous", default_low=vessel_cfg["hu_candidate_low"], default_high=vessel_cfg["hu_candidate_high"]),
        }

    phase_reference_sources = [
        images[mapping.portal_uid],
        images[mapping.arterial_uid],
        images[mapping.venous_uid],
    ]
    union_reference_sources = phase_reference_sources.copy()
    if label_image is not None:
        union_reference_sources.append(label_image)
    reference_grid = build_union_reference_image(
        union_reference_sources,
        base_image=images[mapping.portal_uid],
        pixel_id=images[mapping.portal_uid].GetPixelID(),
    )
    reference = compose_reference_image(
        phase_reference_sources,
        reference_grid,
        default_value=-1024,
        pixel_id=images[mapping.portal_uid].GetPixelID(),
    )
    reference_arr = sitk.GetArrayFromImage(reference).astype(np.float32)
    body_mask = body_region_mask(
        reference_arr,
        spacing_xyz=_spacing_xyz(reference),
        min_hu=vessel_cfg.get("body_min_hu", -600),
        closing_mm=vessel_cfg.get("body_closing_mm", 5),
        dilation_mm=vessel_cfg.get("body_dilation_mm", 2),
    )
    warnings: list[str] = []
    totalseg_cache_path: Path | None = None
    totalseg_priors: TotalSegPriors | None = None
    vessel_anchor_masks: dict[str, np.ndarray] = {}
    vessel_anchor_any = np.zeros(reference_arr.shape, dtype=bool)
    anchor_source = "none"
    if manual_label:
        label_hard_exclusion, soft_penalty = _label_masks(label_image, reference, config)
        hard_exclusion = ~body_mask if label_hard_exclusion is None else (label_hard_exclusion | ~body_mask)
    else:
        totalseg_priors, totalseg_cache_path = _totalseg_priors(
            reference,
            output_dir=output_dir,
            config=config,
            device=totalseg_device,
            force=force_totalseg,
        )
        hard_exclusion = totalseg_priors.hard_exclusion | ~body_mask
        soft_penalty = totalseg_priors.soft_penalty
        vessel_anchor_masks = {
            phase: mask & body_mask & ~totalseg_priors.hard_exclusion
            for phase, mask in totalseg_priors.vessel_anchors_by_phase.items()
        }
        vessel_anchor_any = totalseg_priors.vessel_anchor_any & body_mask & ~totalseg_priors.hard_exclusion
        hu_windows = _estimate_hu_windows_from_images(
            images,
            mapping,
            reference,
            config=config,
            hard_exclusion=hard_exclusion,
            soft_penalty=soft_penalty,
            vessel_anchor_masks=vessel_anchor_masks,
        )
    liver_mask = None
    if not manual_label and totalseg_priors is not None:
        liver_mask = totalseg_priors.named_masks.get("liver")
        if liver_mask is not None:
            liver_mask = liver_mask & body_mask & ~totalseg_priors.hard_exclusion
    phase_cfg = config["phase_detection"]
    ids = phase_cfg["label_ids"]
    vessel_ids = {ids["aorta"], ids["ivc"], ids["portal_vein"], ids["liver_vein"], ids["celiac_artery"]}
    anchors = anchor_mask_from_weak_label(label_image, reference, vessel_label_ids=vessel_ids) if manual_label else None
    anchor_multilabel = None
    if manual_label:
        anchor_multilabel = anchor_multilabel_from_weak_label(
            label_image,
            reference,
            arterial_ids={ids["aorta"], ids["celiac_artery"]},
            portal_ids={ids["portal_vein"]},
            venous_ids={ids["ivc"], ids["liver_vein"]},
        )
    arterial_mask, arterial_conf, arterial_recovery, arterial_recovery_mask = _phase_candidate(
        images[mapping.arterial_uid],
        reference,
        phase_name="arterial",
        config=config,
        hu_window=hu_windows["arterial"],
        vesselness_mode=vesselness_mode,
        anchor_mask=anchors if manual_label else vessel_anchor_masks.get("arterial"),
        hard_exclusion_mask=hard_exclusion,
        soft_penalty_mask=soft_penalty,
        allowed_mask=body_mask,
        liver_mask=liver_mask,
    )
    portal_mask, portal_conf, portal_recovery, portal_recovery_mask = _phase_candidate(
        images[mapping.portal_uid],
        reference,
        phase_name="portal",
        config=config,
        hu_window=hu_windows["portal"],
        vesselness_mode=vesselness_mode,
        anchor_mask=anchors if manual_label else vessel_anchor_masks.get("portal"),
        hard_exclusion_mask=hard_exclusion,
        soft_penalty_mask=soft_penalty,
        allowed_mask=body_mask,
        liver_mask=liver_mask,
    )
    venous_mask, venous_conf, venous_recovery, venous_recovery_mask = _phase_candidate(
        images[mapping.venous_uid],
        reference,
        phase_name="venous",
        config=config,
        hu_window=hu_windows["venous"],
        vesselness_mode=vesselness_mode,
        anchor_mask=anchors if manual_label else vessel_anchor_masks.get("venous"),
        hard_exclusion_mask=hard_exclusion,
        soft_penalty_mask=soft_penalty,
        allowed_mask=body_mask,
        liver_mask=liver_mask,
    )
    fused = fuse_phase_candidates(
        arterial_mask=arterial_mask,
        portal_mask=portal_mask,
        venous_mask=venous_mask,
        confidence_maps={"arterial": arterial_conf, "portal": portal_conf, "venous": venous_conf},
    )
    intrahepatic_recovery_mask = portal_recovery_mask | venous_recovery_mask
    surface_prune = {
        "surface_pruned_voxels": 0,
        "surface_pruned_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    if not manual_label and liver_mask is not None:
        recovery_cfg = vessel_cfg.get("intrahepatic_recovery", {})
        if isinstance(recovery_cfg, dict):
            surface_prune = apply_liver_surface_recovery_gate(
                fused.multilabel,
                fused.confidence,
                recovery_mask=intrahepatic_recovery_mask,
                liver_mask=liver_mask,
                spacing_xyz=_spacing_xyz(reference),
                enabled=bool(recovery_cfg.get("surface_prune_enabled", False)),
                surface_depth_mm=float(recovery_cfg.get("surface_prune_depth_mm", 0.0)),
                confidence_min=float(recovery_cfg.get("surface_prune_confidence_min", 0.75)),
            )
    if anchors is not None:
        anchor_source = "manual_label_legacy"
        vessel_cfg = config["vessel_extraction"]
        anchored_mask = keep_components_near_anchors(fused.multilabel > 0, anchors, dilation_voxels=6)
        anchored_mask = keep_mask_within_anchor_distance(
            anchored_mask,
            anchors,
            spacing_xyz=_spacing_xyz(reference),
            max_distance_mm=vessel_cfg["final_anchor_distance_mm"],
        )
        fused.multilabel[~anchored_mask] = 0
        fused.confidence[~anchored_mask] = 0.0
        if anchor_multilabel is not None:
            fused.multilabel[anchor_multilabel > 0] = anchor_multilabel[anchor_multilabel > 0]
            fused.confidence[anchor_multilabel > 0] = 1.0
    else:
        if vessel_anchor_any.any():
            anchor_source = "totalseg_vessel_priors"
            anchored_mask = keep_components_near_anchors(fused.multilabel > 0, vessel_anchor_any, dilation_voxels=6)
            if intrahepatic_recovery_mask.any():
                anchored_mask |= intrahepatic_recovery_mask
            fused.multilabel[~anchored_mask] = 0
            fused.confidence[~anchored_mask] = 0.0
        else:
            warnings.append("totalseg_vessel_anchor_empty")
            if vessel_cfg.get("no_label_requires_vessel_anchor", True):
                fused.multilabel[:] = 0
                fused.confidence[:] = 0.0
            else:
                auto_anchors = _auto_anchor_mask(
                    fused.multilabel,
                    fused.confidence,
                    spacing_xyz=_spacing_xyz(reference),
                    config=config,
                )
                if auto_anchors.any():
                    anchor_source = "auto_candidate_confidence"
                    anchored_mask = keep_components_near_anchors(fused.multilabel > 0, auto_anchors, dilation_voxels=6)
                    fused.multilabel[~anchored_mask] = 0
                    fused.confidence[~anchored_mask] = 0.0
                else:
                    warnings.append("auto_anchor_empty")
    final_surface_cleanup = {
        "final_liver_surface_cleanup_voxels": 0,
        "final_surface_cleanup_voxels": 0,
        "final_liver_surface_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    portal_relabel = {
        "portal_relabel_voxels": 0,
        "portal_relabel_bridge_voxels": 0,
        "portal_relabel_by_reason": {
            "eligible_venous_liver": 0,
            "missing_portal_coverage": 0,
            "missing_venous_coverage": 0,
            "insufficient_hu_margin": 0,
        },
    }
    deep_liver_cleanup = {
        "deep_liver_cleanup_voxels": 0,
        "deep_liver_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    isolated_blob_cleanup = {
        "isolated_blob_cleanup_voxels": 0,
        "isolated_blob_cleanup_components": 0,
        "isolated_blob_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    apex_surface_cleanup = {
        "apex_surface_cleanup_voxels": 0,
        "apex_surface_cleanup_components": 0,
        "apex_surface_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
        "apex_surface_cleanup_by_region": {"apex": 0, "surface": 0},
    }
    apex_subsurface_cleanup = {
        "apex_subsurface_cleanup_voxels": 0,
        "apex_subsurface_cleanup_components": 0,
        "apex_subsurface_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
        "apex_subsurface_cleanup_by_region": {"apex": 0, "subsurface": 0},
        "apex_subsurface_cleanup_candidate_voxels": 0,
        "apex_subsurface_cleanup_protected_voxels": 0,
    }
    outer_peripheral_cleanup = {
        "outer_peripheral_cleanup_voxels": 0,
        "outer_peripheral_cleanup_components": 0,
        "outer_peripheral_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    smv_portal_bridge_repair = {
        "smv_portal_bridge_repair_voxels": 0,
        "smv_portal_bridge_repair_pairs": 0,
        "smv_portal_bridge_repair_fallback_voxels": 0,
        "smv_portal_bridge_repair_max_gap_mm": 0.0,
        "smv_portal_bridge_repair_evidence_voxels": 0,
        "smv_portal_bridge_repair_morph_fill_voxels": 0,
        "smv_portal_bridge_repair_rejected_pairs": 0,
        "smv_portal_bridge_repair_rejected_by_reason": {
            "insufficient_evidence": 0,
            "excessive_fill": 0,
            "not_connected": 0,
        },
    }
    post_anchor_peripheral_cleanup = {
        "post_anchor_peripheral_cleanup_voxels": 0,
        "post_anchor_peripheral_cleanup_components": 0,
        "post_anchor_peripheral_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
    }
    liver_surface_sheet_cleanup = {
        "liver_surface_sheet_cleanup_voxels": 0,
        "liver_surface_sheet_cleanup_components": 0,
        "liver_surface_sheet_cleanup_by_label": {"arterial": 0, "portal": 0, "venous": 0},
        "liver_surface_sheet_cleanup_protected_voxels": 0,
        "liver_surface_sheet_cleanup_candidate_voxels": 0,
    }
    intrahepatic_trunk_connectivity_before = {
        "intrahepatic_trunk_connected": True,
        "intrahepatic_trunk_components": 0,
        "intrahepatic_trunk_disconnected_components": 0,
        "intrahepatic_trunk_largest_disconnected_volume_mm3": 0.0,
        "intrahepatic_trunk_min_gap_mm": 0.0,
    }
    intrahepatic_trunk_connectivity_after = dict(intrahepatic_trunk_connectivity_before)
    intrahepatic_trunk_reconnect = {
        "intrahepatic_trunk_reconnect_voxels": 0,
        "intrahepatic_trunk_reconnect_pairs": 0,
        "intrahepatic_trunk_reconnect_max_gap_mm": 0.0,
        "intrahepatic_trunk_reconnect_evidence_voxels": 0,
        "intrahepatic_trunk_reconnect_morph_fill_voxels": 0,
        "intrahepatic_trunk_reconnect_rejected_pairs": 0,
        "intrahepatic_trunk_reconnect_rejected_by_reason": {
            "insufficient_evidence": 0,
            "excessive_fill": 0,
            "not_connected": 0,
        },
    }
    hilar_protection_mask = np.zeros(fused.multilabel.shape, dtype=bool)
    hilar_protection_voxels = 0
    smv_portal_protection_mask = np.zeros(fused.multilabel.shape, dtype=bool)
    smv_portal_protection_voxels = 0
    portal_relabel_bridge_mask = np.zeros(fused.multilabel.shape, dtype=bool)
    portal_hu = venous_hu = portal_coverage = venous_coverage = None
    if not manual_label and liver_mask is not None:
        hilar_cfg = vessel_cfg.get("hilar_protection", {})
        if isinstance(hilar_cfg, dict) and bool(hilar_cfg.get("enabled", False)):
            hilar_anchor = np.zeros(fused.multilabel.shape, dtype=bool)
            for phase in ("portal", "venous"):
                phase_anchor = vessel_anchor_masks.get(phase)
                if phase_anchor is not None:
                    hilar_anchor |= phase_anchor.astype(bool, copy=False)
            hilar_protection_mask = build_hilar_protection_mask(
                hilar_anchor,
                spacing_xyz=_spacing_xyz(reference),
                protection_distance_mm=float(hilar_cfg.get("distance_mm", 0.0)),
            ) & liver_mask
            hilar_protection_voxels = int(hilar_protection_mask.sum())

        smv_portal_cfg = vessel_cfg.get("smv_portal_bridge_protection", {})
        if isinstance(smv_portal_cfg, dict) and bool(smv_portal_cfg.get("enabled", False)):
            portal_anchor = vessel_anchor_masks.get("portal")
            if portal_anchor is not None:
                smv_portal_protection_mask = build_smv_portal_protection_mask(
                    portal_anchor,
                    body_mask=body_mask,
                    hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
                    spacing_xyz=_spacing_xyz(reference),
                    protection_distance_mm=float(smv_portal_cfg.get("distance_mm", 0.0)),
                )
                smv_portal_protection_voxels = int(smv_portal_protection_mask.sum())
        combined_protection_mask = hilar_protection_mask | smv_portal_protection_mask

        final_surface_cfg = vessel_cfg.get("final_liver_surface_cleanup", {})
        portal_relabel_cfg = vessel_cfg.get("portal_from_venous_relabel", {})
        smv_bridge_repair_cfg = vessel_cfg.get("smv_portal_bridge_repair", {})
        deep_cleanup_cfg = vessel_cfg.get("deep_liver_cleanup", {})
        blob_cleanup_cfg = vessel_cfg.get("isolated_liver_blob_cleanup", {})
        apex_surface_cleanup_cfg = vessel_cfg.get("apex_surface_morph_cleanup", {})
        outer_peripheral_cleanup_cfg = vessel_cfg.get("outer_peripheral_blob_cleanup", {})
        protected_cleanup_enabled = (
            isinstance(hilar_cfg, dict)
            and bool(hilar_cfg.get("enabled", False))
            and (
                (
                    isinstance(deep_cleanup_cfg, dict)
                    and bool(deep_cleanup_cfg.get("enabled", False))
                )
                or (
                    isinstance(blob_cleanup_cfg, dict)
                    and bool(blob_cleanup_cfg.get("enabled", False))
                )
                or (
                    isinstance(apex_surface_cleanup_cfg, dict)
                    and bool(apex_surface_cleanup_cfg.get("enabled", False))
                )
                or (
                    isinstance(outer_peripheral_cleanup_cfg, dict)
                    and bool(outer_peripheral_cleanup_cfg.get("enabled", False))
                )
            )
        )
        portal_hu_needed = (
            isinstance(portal_relabel_cfg, dict)
            and bool(portal_relabel_cfg.get("enabled", False))
        ) or (
            isinstance(smv_bridge_repair_cfg, dict)
            and bool(smv_bridge_repair_cfg.get("enabled", False))
        )
        if portal_hu_needed:
            portal_hu, portal_coverage = _resampled_hu_and_coverage(images[mapping.portal_uid], reference)
            venous_hu, venous_coverage = _resampled_hu_and_coverage(images[mapping.venous_uid], reference)
        else:
            portal_hu = venous_hu = portal_coverage = venous_coverage = None

        pre_portal_relabel_venous_mask = fused.multilabel == 3
        if (
            isinstance(portal_relabel_cfg, dict)
            and bool(portal_relabel_cfg.get("enabled", False))
            and portal_hu is not None
            and venous_hu is not None
            and portal_coverage is not None
            and venous_coverage is not None
        ):
            portal_relabel = apply_portal_from_venous_relabel(
                fused.multilabel,
                fused.confidence,
                portal_hu=portal_hu,
                venous_hu=venous_hu,
                portal_coverage=portal_coverage,
                venous_coverage=venous_coverage,
                liver_mask=liver_mask | smv_portal_protection_mask,
                enabled=True,
                min_portal_minus_venous_hu=float(portal_relabel_cfg.get("min_portal_minus_venous_hu", 30.0)),
                protection_mask=combined_protection_mask if protected_cleanup_enabled else None,
                protected_min_portal_minus_venous_hu=float(
                    portal_relabel_cfg.get(
                        "protected_min_portal_minus_venous_hu",
                        portal_relabel_cfg.get("min_portal_minus_venous_hu", 30.0),
                    )
                ),
            )
            portal_relabel_bridge_mask = (fused.multilabel == 2) & pre_portal_relabel_venous_mask

        if isinstance(final_surface_cfg, dict) and bool(final_surface_cfg.get("enabled", False)):
            final_surface_cleanup = apply_final_liver_surface_cleanup(
                fused.multilabel,
                fused.confidence,
                liver_mask=liver_mask,
                spacing_xyz=_spacing_xyz(reference),
                enabled=True,
                surface_depth_mm=float(final_surface_cfg.get("surface_depth_mm", 0.0)),
                confidence_min=float(final_surface_cfg.get("confidence_min", 0.78)),
                protection_mask=hilar_protection_mask if protected_cleanup_enabled else None,
            )

        anchor_output = inject_totalseg_vessel_anchors(
            fused.multilabel,
            fused.confidence,
            anchors_by_phase=vessel_anchor_masks,
            body_mask=body_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            liver_mask=liver_mask,
            enabled=(not manual_label and bool(vessel_cfg.get("include_totalseg_vessel_anchors_in_output", False))),
            include_liver=bool(vessel_cfg.get("totalseg_vessel_anchors_include_liver", False)),
        )

    pre_bridge_portal_mask = fused.multilabel == 2
    bridge_repaired_mask = np.zeros(fused.multilabel.shape, dtype=bool)
    if (
        not manual_label
        and isinstance(vessel_cfg.get("smv_portal_bridge_repair", {}), dict)
        and bool(vessel_cfg.get("smv_portal_bridge_repair", {}).get("enabled", False))
        and portal_hu is not None
        and venous_hu is not None
        and portal_coverage is not None
        and venous_coverage is not None
    ):
        smv_bridge_repair_cfg = vessel_cfg["smv_portal_bridge_repair"]
        assert isinstance(smv_bridge_repair_cfg, dict)
        smv_portal_bridge_repair = apply_smv_portal_bridge_repair(
            fused.multilabel,
            fused.confidence,
            body_mask=body_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            protection_mask=smv_portal_protection_mask,
            portal_hu=portal_hu,
            venous_hu=venous_hu,
            portal_coverage=portal_coverage,
            venous_coverage=venous_coverage,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            max_gap_mm=float(smv_bridge_repair_cfg.get("max_gap_mm", 0.0)),
            corridor_radius_mm=float(smv_bridge_repair_cfg.get("corridor_radius_mm", 0.0)),
            endpoint_min_volume_mm3=float(smv_bridge_repair_cfg.get("endpoint_min_volume_mm3", 0.0)),
            min_portal_minus_venous_hu=float(smv_bridge_repair_cfg.get("min_portal_minus_venous_hu", 10.0)),
            fallback_centerline_enabled=bool(smv_bridge_repair_cfg.get("fallback_centerline_enabled", False)),
            bridge_confidence=float(smv_bridge_repair_cfg.get("bridge_confidence", 0.85)),
            morphological_tube_fill_enabled=bool(smv_bridge_repair_cfg.get("morphological_tube_fill_enabled", False)),
            tube_radius_mm=float(smv_bridge_repair_cfg.get("tube_radius_mm", 0.0)),
            closing_radius_mm=float(smv_bridge_repair_cfg.get("closing_radius_mm", 0.0)),
            min_evidence_fraction=float(smv_bridge_repair_cfg.get("min_evidence_fraction", 0.0)),
            max_fill_to_evidence_ratio=float(smv_bridge_repair_cfg.get("max_fill_to_evidence_ratio", float("inf"))),
        )
        bridge_repaired_mask = (fused.multilabel == 2) & ~pre_bridge_portal_mask

    bridge_seed_mask = bridge_repaired_mask | portal_relabel_bridge_mask
    trunk_seed_mask = bridge_seed_mask | hilar_protection_mask | smv_portal_protection_mask
    if (
        not manual_label
        and liver_mask is not None
        and trunk_seed_mask.any()
        and isinstance(vessel_cfg.get("intrahepatic_trunk_reconnect", {}), dict)
        and bool(vessel_cfg.get("intrahepatic_trunk_reconnect", {}).get("enabled", False))
    ):
        trunk_reconnect_cfg = vessel_cfg["intrahepatic_trunk_reconnect"]
        assert isinstance(trunk_reconnect_cfg, dict)
        trunk_target_labels = trunk_reconnect_cfg.get("target_labels", ["portal", "venous"])
        candidate_mask = portal_mask | venous_mask | intrahepatic_recovery_mask
        intrahepatic_trunk_connectivity_before = measure_intrahepatic_trunk_connectivity(
            fused.multilabel,
            trunk_seed_mask=trunk_seed_mask,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            target_labels=trunk_target_labels,
            min_component_volume_mm3=float(trunk_reconnect_cfg.get("min_component_volume_mm3", 0.0)),
        )
        intrahepatic_trunk_reconnect = apply_intrahepatic_trunk_reconnect(
            fused.multilabel,
            fused.confidence,
            trunk_seed_mask=trunk_seed_mask,
            candidate_mask=candidate_mask,
            liver_mask=liver_mask,
            body_mask=body_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            target_labels=trunk_target_labels,
            max_gap_mm=float(trunk_reconnect_cfg.get("max_gap_mm", 0.0)),
            corridor_radius_mm=float(trunk_reconnect_cfg.get("corridor_radius_mm", 0.0)),
            tube_radius_mm=float(trunk_reconnect_cfg.get("tube_radius_mm", 0.0)),
            closing_radius_mm=float(trunk_reconnect_cfg.get("closing_radius_mm", 0.0)),
            min_component_volume_mm3=float(trunk_reconnect_cfg.get("min_component_volume_mm3", 0.0)),
            min_evidence_fraction=float(trunk_reconnect_cfg.get("min_evidence_fraction", 0.0)),
            max_fill_to_evidence_ratio=float(trunk_reconnect_cfg.get("max_fill_to_evidence_ratio", float("inf"))),
            bridge_confidence=float(trunk_reconnect_cfg.get("bridge_confidence", 0.86)),
        )
        intrahepatic_trunk_connectivity_after = measure_intrahepatic_trunk_connectivity(
            fused.multilabel,
            trunk_seed_mask=trunk_seed_mask,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            target_labels=trunk_target_labels,
            min_component_volume_mm3=float(trunk_reconnect_cfg.get("min_component_volume_mm3", 0.0)),
        )

    deep_anchor = np.zeros(fused.multilabel.shape, dtype=bool)
    for phase in ("portal", "venous"):
        phase_anchor = vessel_anchor_masks.get(phase)
        if phase_anchor is not None:
            deep_anchor |= phase_anchor.astype(bool, copy=False)
    protected_trunk_mask = np.zeros(fused.multilabel.shape, dtype=bool)
    if trunk_seed_mask.any():
        protected_trunk_mask = keep_components_near_anchors(
            fused.multilabel > 0,
            trunk_seed_mask,
            dilation_voxels=4,
        )
        if protected_trunk_mask.any():
            protected_trunk_mask &= body_mask
            if liver_mask is not None:
                protected_trunk_mask &= liver_mask

    combined_apex_protection_mask = bridge_seed_mask | protected_trunk_mask
    apex_protection_mask = combined_apex_protection_mask if combined_apex_protection_mask.any() else None
    apex_subsurface_protection_mask = apex_protection_mask

    if isinstance(deep_cleanup_cfg, dict) and bool(deep_cleanup_cfg.get("enabled", False)):
        deep_liver_cleanup = apply_deep_liver_cleanup(
            fused.multilabel,
            fused.confidence,
            anchor_mask=deep_anchor,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            min_anchor_distance_mm=float(deep_cleanup_cfg.get("min_anchor_distance_mm", 0.0)),
            confidence_min=float(deep_cleanup_cfg.get("confidence_min", 0.78)),
        )
    elif isinstance(blob_cleanup_cfg, dict) and bool(blob_cleanup_cfg.get("enabled", False)):
        isolated_blob_cleanup = apply_isolated_liver_blob_cleanup(
            fused.multilabel,
            fused.confidence,
            anchor_mask=deep_anchor,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            max_component_volume_mm3=float(blob_cleanup_cfg.get("max_component_volume_mm3", 0.0)),
            max_component_elongation=float(blob_cleanup_cfg.get("max_component_elongation", 2.0)),
            confidence_min=float(blob_cleanup_cfg.get("confidence_min", 0.78)),
            anchor_dilation_mm=float(blob_cleanup_cfg.get("anchor_dilation_mm", 0.0)),
            protection_mask=hilar_protection_mask if protected_cleanup_enabled else None,
        )
    if isinstance(apex_surface_cleanup_cfg, dict) and bool(apex_surface_cleanup_cfg.get("enabled", False)):
        apex_surface_cleanup = apply_apex_surface_morph_cleanup(
            fused.multilabel,
            fused.confidence,
            anchor_mask=deep_anchor,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            surface_depth_mm=float(apex_surface_cleanup_cfg.get("surface_depth_mm", 0.0)),
            apex_fraction=float(apex_surface_cleanup_cfg.get("apex_fraction", 0.0)),
            confidence_min=float(apex_surface_cleanup_cfg.get("confidence_min", 0.82)),
            max_component_volume_mm3=float(apex_surface_cleanup_cfg.get("max_component_volume_mm3", 0.0)),
            max_component_elongation=float(apex_surface_cleanup_cfg.get("max_component_elongation", 2.4)),
            anchor_dilation_mm=float(apex_surface_cleanup_cfg.get("anchor_dilation_mm", 0.0)),
            protection_mask=apex_protection_mask,
        )
    if (
        isinstance(vessel_cfg.get("apex_subsurface_cleanup", {}), dict)
        and bool(vessel_cfg.get("apex_subsurface_cleanup", {}).get("enabled", False))
    ):
        apex_subsurface_cfg = vessel_cfg["apex_subsurface_cleanup"]
        assert isinstance(apex_subsurface_cfg, dict)
        apex_subsurface_cleanup = apply_apex_subsurface_cleanup(
            fused.multilabel,
            fused.confidence,
            anchor_mask=deep_anchor,
            liver_mask=liver_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            apex_fraction=float(apex_subsurface_cfg.get("apex_fraction", 0.12)),
            subsurface_min_depth_mm=float(apex_subsurface_cfg.get("subsurface_min_depth_mm", 8.0)),
            subsurface_max_depth_mm=float(apex_subsurface_cfg.get("subsurface_max_depth_mm", 15.0)),
            confidence_min=float(apex_subsurface_cfg.get("confidence_min", 0.80)),
            min_component_volume_mm3=float(apex_subsurface_cfg.get("min_component_volume_mm3", 120.0)),
            max_component_volume_mm3=float(apex_subsurface_cfg.get("max_component_volume_mm3", 1500.0)),
            max_component_linearity=float(apex_subsurface_cfg.get("max_component_linearity", 4.5)),
            min_surface_fraction=float(apex_subsurface_cfg.get("min_surface_fraction", 0.35)),
            anchor_dilation_mm=float(apex_subsurface_cfg.get("anchor_dilation_mm", 0.0)),
            protection_mask=apex_subsurface_protection_mask,
        )
    if isinstance(outer_peripheral_cleanup_cfg, dict) and bool(outer_peripheral_cleanup_cfg.get("enabled", False)):
        outer_peripheral_cleanup = apply_outer_peripheral_blob_cleanup(
            fused.multilabel,
            fused.confidence,
            body_mask=body_mask,
            liver_mask=liver_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            protection_mask=combined_protection_mask,
            anchor_mask=deep_anchor,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            max_component_volume_mm3=float(outer_peripheral_cleanup_cfg.get("max_component_volume_mm3", 0.0)),
            max_component_linearity=float(outer_peripheral_cleanup_cfg.get("max_component_linearity", 2.2)),
            confidence_min=float(outer_peripheral_cleanup_cfg.get("confidence_min", 0.82)),
            anchor_dilation_mm=float(outer_peripheral_cleanup_cfg.get("anchor_dilation_mm", 0.0)),
        )
    if (
        not manual_label
        and totalseg_priors is not None
        and isinstance(vessel_cfg.get("post_anchor_peripheral_component_audit", {}), dict)
        and bool(vessel_cfg.get("post_anchor_peripheral_component_audit", {}).get("enabled", False))
    ):
        post_anchor_cfg = vessel_cfg["post_anchor_peripheral_component_audit"]
        assert isinstance(post_anchor_cfg, dict)
        envelope_names = post_anchor_cfg.get("organ_envelope_masks", [])
        envelope_seed = np.zeros(fused.multilabel.shape, dtype=bool)
        if isinstance(envelope_names, list):
            for name in envelope_names:
                mask = totalseg_priors.named_masks.get(str(name))
                if mask is not None:
                    envelope_seed |= mask.astype(bool, copy=False)
        anchor_envelope = np.zeros(fused.multilabel.shape, dtype=bool)
        if envelope_seed.any():
            sampling_zyx = (
                float(_spacing_xyz(reference)[2]),
                float(_spacing_xyz(reference)[1]),
                float(_spacing_xyz(reference)[0]),
            )
            envelope_distance = ndi.distance_transform_edt(~(envelope_seed & body_mask), sampling=sampling_zyx)
            anchor_envelope = envelope_distance <= float(post_anchor_cfg.get("organ_envelope_dilation_mm", 0.0))
        core_anchor = (vessel_anchor_any & anchor_envelope) | bridge_repaired_mask
        post_anchor_peripheral_cleanup = apply_post_anchor_peripheral_component_audit(
            fused.multilabel,
            fused.confidence,
            body_mask=body_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            envelope_seed_mask=envelope_seed & body_mask,
            core_anchor_mask=core_anchor & body_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            organ_envelope_dilation_mm=float(post_anchor_cfg.get("organ_envelope_dilation_mm", 0.0)),
            core_anchor_protection_mm=float(post_anchor_cfg.get("core_anchor_protection_mm", 0.0)),
            min_component_volume_mm3=float(post_anchor_cfg.get("min_component_volume_mm3", 0.0)),
            max_component_linearity=float(post_anchor_cfg.get("max_component_linearity", 3.0)),
            confidence_max=float(post_anchor_cfg.get("confidence_max", 1.01)),
        )
    if (
        not manual_label
        and liver_mask is not None
        and isinstance(vessel_cfg.get("liver_surface_sheet_cleanup", {}), dict)
        and bool(vessel_cfg.get("liver_surface_sheet_cleanup", {}).get("enabled", False))
    ):
        sheet_cfg = vessel_cfg["liver_surface_sheet_cleanup"]
        assert isinstance(sheet_cfg, dict)
        surface_core_anchor = vessel_anchor_any & body_mask
        liver_surface_sheet_cleanup = apply_liver_surface_sheet_cleanup(
            fused.multilabel,
            fused.confidence,
            liver_mask=liver_mask,
            body_mask=body_mask,
            hard_exclusion_mask=hard_exclusion if hard_exclusion is not None else np.zeros(body_mask.shape, dtype=bool),
            core_anchor_mask=surface_core_anchor,
            bridge_mask=bridge_seed_mask if bridge_seed_mask.any() else bridge_repaired_mask,
            spacing_xyz=_spacing_xyz(reference),
            enabled=True,
            surface_depth_mm=float(sheet_cfg.get("surface_depth_mm", 0.0)),
            min_component_volume_mm3=float(sheet_cfg.get("min_component_volume_mm3", 0.0)),
            max_component_volume_mm3=float(sheet_cfg.get("max_component_volume_mm3", 0.0)),
            max_component_linearity=float(sheet_cfg.get("max_component_linearity", 0.0)),
            min_surface_fraction=float(sheet_cfg.get("min_surface_fraction", 0.0)),
            confidence_max=float(sheet_cfg.get("confidence_max", 1.01)),
            core_anchor_protection_mm=float(sheet_cfg.get("core_anchor_protection_mm", 0.0)),
            bridge_protection_mm=float(sheet_cfg.get("bridge_protection_mm", 0.0)),
            target_labels=sheet_cfg.get("target_labels", ["arterial", "portal", "venous"]),
        )
    table_removed_components, table_removed_voxels = _remove_table_like_components(
        fused.multilabel,
        fused.confidence,
        enabled=vessel_cfg.get("reject_table_like_components", True),
    )
    quality_metrics = {
        "body_mask_voxels": int(body_mask.sum()),
        "hard_exclusion_voxels": int(hard_exclusion.sum()) if hard_exclusion is not None else 0,
        "vessel_anchor_voxels": int(vessel_anchor_any.sum() if not manual_label else anchors.sum() if anchors is not None else 0),
        "intrahepatic_recovery_voxels": int(portal_recovery["kept_voxels"] + venous_recovery["kept_voxels"]),
        "intrahepatic_recovery_components": int(portal_recovery["kept_components"] + venous_recovery["kept_components"]),
        "intrahepatic_recovery_by_phase": {
            "arterial": arterial_recovery,
            "portal": portal_recovery,
            "venous": venous_recovery,
        },
        "intrahepatic_surface_pruned_voxels": int(surface_prune["surface_pruned_voxels"]),
        "intrahepatic_surface_pruned_by_label": surface_prune["surface_pruned_by_label"],
        "hilar_protection_voxels": int(hilar_protection_voxels),
        "smv_portal_protection_voxels": int(smv_portal_protection_voxels),
        "final_liver_surface_cleanup_voxels": int(final_surface_cleanup["final_liver_surface_cleanup_voxels"]),
        "final_surface_cleanup_voxels": int(final_surface_cleanup["final_surface_cleanup_voxels"]),
        "final_liver_surface_cleanup_by_label": final_surface_cleanup["final_liver_surface_cleanup_by_label"],
        "portal_relabel_voxels": int(portal_relabel["portal_relabel_voxels"]),
        "portal_relabel_bridge_voxels": int(portal_relabel["portal_relabel_bridge_voxels"]),
        "portal_relabel_by_reason": portal_relabel["portal_relabel_by_reason"],
        "deep_liver_cleanup_voxels": int(deep_liver_cleanup["deep_liver_cleanup_voxels"]),
        "deep_liver_cleanup_by_label": deep_liver_cleanup["deep_liver_cleanup_by_label"],
        "isolated_blob_cleanup_voxels": int(isolated_blob_cleanup["isolated_blob_cleanup_voxels"]),
        "isolated_blob_cleanup_components": int(isolated_blob_cleanup["isolated_blob_cleanup_components"]),
        "isolated_blob_cleanup_by_label": isolated_blob_cleanup["isolated_blob_cleanup_by_label"],
        "apex_surface_cleanup_voxels": int(apex_surface_cleanup["apex_surface_cleanup_voxels"]),
        "apex_surface_cleanup_components": int(apex_surface_cleanup["apex_surface_cleanup_components"]),
        "apex_surface_cleanup_by_label": apex_surface_cleanup["apex_surface_cleanup_by_label"],
        "apex_surface_cleanup_by_region": apex_surface_cleanup["apex_surface_cleanup_by_region"],
        "apex_subsurface_cleanup_voxels": int(apex_subsurface_cleanup["apex_subsurface_cleanup_voxels"]),
        "apex_subsurface_cleanup_components": int(apex_subsurface_cleanup["apex_subsurface_cleanup_components"]),
        "apex_subsurface_cleanup_by_label": apex_subsurface_cleanup["apex_subsurface_cleanup_by_label"],
        "apex_subsurface_cleanup_by_region": apex_subsurface_cleanup["apex_subsurface_cleanup_by_region"],
        "apex_subsurface_cleanup_candidate_voxels": int(apex_subsurface_cleanup["apex_subsurface_cleanup_candidate_voxels"]),
        "apex_subsurface_cleanup_protected_voxels": int(apex_subsurface_cleanup["apex_subsurface_cleanup_protected_voxels"]),
        "protected_trunk_voxels": int(protected_trunk_mask.sum()),
        "outer_peripheral_cleanup_voxels": int(outer_peripheral_cleanup["outer_peripheral_cleanup_voxels"]),
        "outer_peripheral_cleanup_components": int(outer_peripheral_cleanup["outer_peripheral_cleanup_components"]),
        "outer_peripheral_cleanup_by_label": outer_peripheral_cleanup["outer_peripheral_cleanup_by_label"],
        "smv_portal_bridge_repair_voxels": int(smv_portal_bridge_repair["smv_portal_bridge_repair_voxels"]),
        "smv_portal_bridge_repair_pairs": int(smv_portal_bridge_repair["smv_portal_bridge_repair_pairs"]),
        "smv_portal_bridge_repair_fallback_voxels": int(smv_portal_bridge_repair["smv_portal_bridge_repair_fallback_voxels"]),
        "smv_portal_bridge_repair_max_gap_mm": float(smv_portal_bridge_repair["smv_portal_bridge_repair_max_gap_mm"]),
        "smv_portal_bridge_repair_evidence_voxels": int(smv_portal_bridge_repair["smv_portal_bridge_repair_evidence_voxels"]),
        "smv_portal_bridge_repair_morph_fill_voxels": int(smv_portal_bridge_repair["smv_portal_bridge_repair_morph_fill_voxels"]),
        "smv_portal_bridge_repair_rejected_pairs": int(smv_portal_bridge_repair["smv_portal_bridge_repair_rejected_pairs"]),
        "smv_portal_bridge_repair_rejected_by_reason": smv_portal_bridge_repair["smv_portal_bridge_repair_rejected_by_reason"],
        "post_anchor_peripheral_cleanup_voxels": int(post_anchor_peripheral_cleanup["post_anchor_peripheral_cleanup_voxels"]),
        "post_anchor_peripheral_cleanup_components": int(post_anchor_peripheral_cleanup["post_anchor_peripheral_cleanup_components"]),
        "post_anchor_peripheral_cleanup_by_label": post_anchor_peripheral_cleanup["post_anchor_peripheral_cleanup_by_label"],
        "liver_surface_sheet_cleanup_voxels": int(liver_surface_sheet_cleanup["liver_surface_sheet_cleanup_voxels"]),
        "liver_surface_sheet_cleanup_components": int(liver_surface_sheet_cleanup["liver_surface_sheet_cleanup_components"]),
        "liver_surface_sheet_cleanup_by_label": liver_surface_sheet_cleanup["liver_surface_sheet_cleanup_by_label"],
        "liver_surface_sheet_cleanup_protected_voxels": int(liver_surface_sheet_cleanup["liver_surface_sheet_cleanup_protected_voxels"]),
        "liver_surface_sheet_cleanup_candidate_voxels": int(liver_surface_sheet_cleanup["liver_surface_sheet_cleanup_candidate_voxels"]),
        "intrahepatic_trunk_connected_before": bool(intrahepatic_trunk_connectivity_before["intrahepatic_trunk_connected"]),
        "intrahepatic_trunk_connected_after": bool(intrahepatic_trunk_connectivity_after["intrahepatic_trunk_connected"]),
        "intrahepatic_trunk_disconnected_components_before": int(
            intrahepatic_trunk_connectivity_before["intrahepatic_trunk_disconnected_components"]
        ),
        "intrahepatic_trunk_disconnected_components_after": int(
            intrahepatic_trunk_connectivity_after["intrahepatic_trunk_disconnected_components"]
        ),
        "intrahepatic_trunk_min_gap_mm_before": float(intrahepatic_trunk_connectivity_before["intrahepatic_trunk_min_gap_mm"]),
        "intrahepatic_trunk_min_gap_mm_after": float(intrahepatic_trunk_connectivity_after["intrahepatic_trunk_min_gap_mm"]),
        "intrahepatic_trunk_reconnect_voxels": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_voxels"]),
        "intrahepatic_trunk_reconnect_pairs": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_pairs"]),
        "intrahepatic_trunk_reconnect_max_gap_mm": float(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_max_gap_mm"]),
        "intrahepatic_trunk_reconnect_evidence_voxels": int(
            intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_evidence_voxels"]
        ),
        "intrahepatic_trunk_reconnect_morph_fill_voxels": int(
            intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_morph_fill_voxels"]
        ),
        "intrahepatic_trunk_reconnect_rejected_pairs": int(
            intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_rejected_pairs"]
        ),
        "intrahepatic_trunk_reconnect_rejected_by_reason": intrahepatic_trunk_reconnect[
            "intrahepatic_trunk_reconnect_rejected_by_reason"
        ],
        "totalseg_anchor_output_voxels": int(anchor_output["totalseg_anchor_output_voxels"]),
        "totalseg_anchor_output_by_phase": anchor_output["totalseg_anchor_output_by_phase"],
        "outside_body_voxels": int(((fused.multilabel > 0) & ~body_mask).sum()),
        "components_removed_by_table_gate": int(table_removed_components),
        "voxels_removed_by_table_gate": int(table_removed_voxels),
    }
    output_summary = _save_outputs(
        output_dir=output_dir,
        reference=reference,
        reference_arr=reference_arr,
        arterial_mask=arterial_mask,
        portal_mask=portal_mask,
        venous_mask=venous_mask,
        multilabel=fused.multilabel,
        confidence=fused.confidence,
        config=config,
        skip_mesh=skip_mesh,
        quality_metrics=quality_metrics,
    )
    summary = {
        "input": input_path,
        "output_dir": output_dir,
        "label": label_found,
        "candidate_series": selected_candidates,
        "phase_scores": scores,
        "phase_mapping": mapping,
        "guidance_source": "manual_label_legacy" if manual_label else "auto_totalseg_priors",
        "phase_selection_source": "manual_label_roi_hu" if manual_label else "metadata_order",
        "totalseg_cache_path": totalseg_cache_path,
        "hu_windows": hu_windows,
        "vesselness_mode": vesselness_mode,
        "warnings": warnings,
        "anchor_source": anchor_source,
        **output_summary,
    }
    _write_json(output_dir / "metrics_report.json", summary)
    return summary
