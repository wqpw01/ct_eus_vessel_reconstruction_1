from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


PHASE_LABEL_IDS = {
    "arterial": 1,
    "portal": 2,
    "venous": 3,
}

LABEL_NAMES = {
    1: "arterial",
    2: "portal",
    3: "venous",
}


def _label_ids_from_names(names: tuple[str, ...] | list[str]) -> set[int]:
    return {
        PHASE_LABEL_IDS[name]
        for name in names
        if isinstance(name, str) and name in PHASE_LABEL_IDS
    }


def _zero_anchor_metrics() -> dict[str, Any]:
    return {
        "totalseg_anchor_output_voxels": 0,
        "totalseg_anchor_output_by_phase": {
            phase: {"candidate_voxels": 0, "injected_voxels": 0}
            for phase in PHASE_LABEL_IDS
        },
    }


def inject_totalseg_vessel_anchors(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    anchors_by_phase: dict[str, np.ndarray],
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    liver_mask: np.ndarray | None,
    enabled: bool,
    include_liver: bool,
) -> dict[str, Any]:
    metrics = _zero_anchor_metrics()
    if not enabled:
        return metrics

    allowed = body_mask.astype(bool, copy=False) & ~hard_exclusion_mask.astype(bool, copy=False)
    if liver_mask is not None and not include_liver:
        allowed &= ~liver_mask.astype(bool, copy=False)

    total_output = np.zeros(multilabel.shape, dtype=bool)
    for phase, label_id in PHASE_LABEL_IDS.items():
        anchor = anchors_by_phase.get(phase)
        if anchor is None:
            continue
        safe_anchor = anchor.astype(bool, copy=False) & allowed
        changed = safe_anchor & (multilabel != label_id)
        multilabel[safe_anchor] = label_id
        confidence[safe_anchor] = 1.0
        total_output |= safe_anchor
        metrics["totalseg_anchor_output_by_phase"][phase] = {
            "candidate_voxels": int(safe_anchor.sum()),
            "injected_voxels": int(changed.sum()),
        }

    metrics["totalseg_anchor_output_voxels"] = int(total_output.sum())
    return metrics


def apply_liver_surface_recovery_gate(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    recovery_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    surface_depth_mm: float,
    confidence_min: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "surface_pruned_voxels": 0,
        "surface_pruned_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }
    if not enabled or surface_depth_mm <= 0:
        return metrics

    liver = liver_mask.astype(bool, copy=False)
    if not liver.any():
        return metrics

    sampling_zyx = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    depth = ndi.distance_transform_edt(liver, sampling=sampling_zyx)
    prune = (
        (multilabel > 0)
        & recovery_mask.astype(bool, copy=False)
        & liver
        & (depth <= float(surface_depth_mm))
        & (confidence < float(confidence_min))
    )
    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["surface_pruned_voxels"] = int(prune.sum())
    for label_id, name in LABEL_NAMES.items():
        metrics["surface_pruned_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def _zero_final_surface_cleanup_metrics() -> dict[str, Any]:
    return {
        "final_liver_surface_cleanup_voxels": 0,
        "final_surface_cleanup_voxels": 0,
        "final_liver_surface_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }


def build_hilar_protection_mask(
    anchor_mask: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    protection_distance_mm: float,
) -> np.ndarray:
    anchors = anchor_mask.astype(bool, copy=False)
    if protection_distance_mm <= 0 or not anchors.any():
        return np.zeros(anchor_mask.shape, dtype=bool)
    sampling_zyx = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    distance = ndi.distance_transform_edt(~anchors, sampling=sampling_zyx)
    return distance <= float(protection_distance_mm)


def build_smv_portal_protection_mask(
    anchor_mask: np.ndarray,
    *,
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    protection_distance_mm: float,
) -> np.ndarray:
    anchors = anchor_mask.astype(bool, copy=False)
    allowed = body_mask.astype(bool, copy=False) & ~hard_exclusion_mask.astype(bool, copy=False)
    if protection_distance_mm <= 0 or not anchors.any():
        return np.zeros(anchor_mask.shape, dtype=bool)
    distance = ndi.distance_transform_edt(~anchors, sampling=_sampling_zyx(spacing_xyz))
    return (distance <= float(protection_distance_mm)) & allowed


def apply_final_liver_surface_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    surface_depth_mm: float,
    confidence_min: float,
    protection_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = _zero_final_surface_cleanup_metrics()
    if not enabled or surface_depth_mm <= 0:
        return metrics

    liver = liver_mask.astype(bool, copy=False)
    if not liver.any():
        return metrics

    sampling_zyx = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    depth = ndi.distance_transform_edt(liver, sampling=sampling_zyx)
    prune = (
        (multilabel == PHASE_LABEL_IDS["venous"])
        & liver
        & (depth <= float(surface_depth_mm))
        & (confidence < float(confidence_min))
    )
    if protection_mask is not None:
        prune &= ~protection_mask.astype(bool, copy=False)
    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["final_liver_surface_cleanup_voxels"] = int(prune.sum())
    metrics["final_surface_cleanup_voxels"] = int(prune.sum())
    for label_id, name in LABEL_NAMES.items():
        metrics["final_liver_surface_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def _zero_portal_relabel_metrics() -> dict[str, Any]:
    return {
        "portal_relabel_voxels": 0,
        "portal_relabel_bridge_voxels": 0,
        "portal_relabel_by_reason": {
            "eligible_venous_liver": 0,
            "missing_portal_coverage": 0,
            "missing_venous_coverage": 0,
            "insufficient_hu_margin": 0,
        },
    }


def apply_portal_from_venous_relabel(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    portal_hu: np.ndarray,
    venous_hu: np.ndarray,
    portal_coverage: np.ndarray,
    venous_coverage: np.ndarray,
    liver_mask: np.ndarray,
    enabled: bool,
    min_portal_minus_venous_hu: float,
    protection_mask: np.ndarray | None = None,
    protected_min_portal_minus_venous_hu: float | None = None,
) -> dict[str, Any]:
    metrics = _zero_portal_relabel_metrics()
    if not enabled:
        return metrics

    liver = liver_mask.astype(bool, copy=False)
    eligible = (multilabel == PHASE_LABEL_IDS["venous"]) & liver
    metrics["portal_relabel_by_reason"]["eligible_venous_liver"] = int(eligible.sum())
    if not eligible.any():
        return metrics

    portal_valid = portal_coverage.astype(bool, copy=False)
    venous_valid = venous_coverage.astype(bool, copy=False)
    missing_portal = eligible & ~portal_valid
    has_portal = eligible & portal_valid
    missing_venous = has_portal & ~venous_valid
    has_both = has_portal & venous_valid
    hu_delta = portal_hu.astype(np.float32, copy=False) - venous_hu.astype(np.float32, copy=False)
    threshold = np.full(multilabel.shape, float(min_portal_minus_venous_hu), dtype=np.float32)
    protected = np.zeros(multilabel.shape, dtype=bool)
    if protection_mask is not None and protected_min_portal_minus_venous_hu is not None:
        protected = protection_mask.astype(bool, copy=False)
        threshold[protected] = float(protected_min_portal_minus_venous_hu)
    insufficient = has_both & (hu_delta < threshold)
    relabel = has_both & (hu_delta >= threshold)

    multilabel[relabel] = PHASE_LABEL_IDS["portal"]
    metrics["portal_relabel_voxels"] = int(relabel.sum())
    metrics["portal_relabel_bridge_voxels"] = int((relabel & protected).sum())
    metrics["portal_relabel_by_reason"]["missing_portal_coverage"] = int(missing_portal.sum())
    metrics["portal_relabel_by_reason"]["missing_venous_coverage"] = int(missing_venous.sum())
    metrics["portal_relabel_by_reason"]["insufficient_hu_margin"] = int(insufficient.sum())
    return metrics


def _zero_deep_liver_cleanup_metrics() -> dict[str, Any]:
    return {
        "deep_liver_cleanup_voxels": 0,
        "deep_liver_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }


def apply_deep_liver_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    min_anchor_distance_mm: float,
    confidence_min: float,
) -> dict[str, Any]:
    metrics = _zero_deep_liver_cleanup_metrics()
    anchors = anchor_mask.astype(bool, copy=False)
    liver = liver_mask.astype(bool, copy=False)
    if not enabled or min_anchor_distance_mm <= 0 or not anchors.any() or not liver.any():
        return metrics

    sampling_zyx = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    anchor_distance = ndi.distance_transform_edt(~anchors, sampling=sampling_zyx)
    prune = (
        (multilabel == PHASE_LABEL_IDS["venous"])
        & liver
        & (anchor_distance > float(min_anchor_distance_mm))
        & (confidence < float(confidence_min))
    )
    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["deep_liver_cleanup_voxels"] = int(prune.sum())
    for label_id, name in LABEL_NAMES.items():
        metrics["deep_liver_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def _zero_isolated_blob_cleanup_metrics() -> dict[str, Any]:
    return {
        "isolated_blob_cleanup_voxels": 0,
        "isolated_blob_cleanup_components": 0,
        "isolated_blob_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }


def _component_elongation(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    positive = extent[extent > 0]
    return float(positive.max() / max(positive.min(), 1))


def _component_linearity(mask: np.ndarray) -> float:
    coords = np.argwhere(mask).astype(np.float32)
    if coords.shape[0] < 3:
        return float("inf") if coords.shape[0] > 1 else 0.0
    coords -= coords.mean(axis=0, keepdims=True)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    if eigenvalues[0] <= 0:
        return 0.0
    if eigenvalues[1] <= 1e-6:
        return float("inf")
    return float(np.sqrt(eigenvalues[0] / eigenvalues[1]))


def _iter_labeled_component_views(labeled: np.ndarray, count: int):
    component_slices = ndi.find_objects(labeled, max_label=count)
    for component_id, bbox in enumerate(component_slices, start=1):
        if bbox is None:
            continue
        labeled_view = labeled[bbox]
        component_view = labeled_view == component_id
        if component_view.any():
            yield bbox, component_view


def _sampling_zyx(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )


def _anchor_zone(
    anchor_mask: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    dilation_mm: float,
) -> np.ndarray:
    anchors = anchor_mask.astype(bool, copy=False)
    if anchors.any() and dilation_mm > 0:
        anchor_distance = ndi.distance_transform_edt(~anchors, sampling=_sampling_zyx(spacing_xyz))
        return anchor_distance <= float(dilation_mm)
    return anchors


def _zero_apex_surface_cleanup_metrics() -> dict[str, Any]:
    return {
        "apex_surface_cleanup_voxels": 0,
        "apex_surface_cleanup_components": 0,
        "apex_surface_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
        "apex_surface_cleanup_by_region": {
            "apex": 0,
            "surface": 0,
        },
    }


def _liver_apex_mask(liver: np.ndarray, *, apex_fraction: float) -> np.ndarray:
    if apex_fraction <= 0 or not liver.any():
        return np.zeros(liver.shape, dtype=bool)
    z_coords = np.flatnonzero(liver.any(axis=(1, 2)))
    if z_coords.size == 0:
        return np.zeros(liver.shape, dtype=bool)
    z_min = int(z_coords.min())
    z_max = int(z_coords.max())
    z_span = z_max - z_min + 1
    apex_slices = max(1, int(np.ceil(z_span * float(apex_fraction))))
    apex_start = max(z_min, z_max - apex_slices + 1)
    apex = np.zeros(liver.shape, dtype=bool)
    apex[apex_start : z_max + 1] = True
    return apex & liver


def _surface_band_from_mask(mask: np.ndarray, *, spacing_xyz: tuple[float, float, float], depth_mm: float) -> np.ndarray:
    source = mask.astype(bool, copy=False)
    if depth_mm <= 0 or not source.any():
        return np.zeros(source.shape, dtype=bool)
    depth = ndi.distance_transform_edt(source, sampling=_sampling_zyx(spacing_xyz))
    return source & (depth <= float(depth_mm))


def apply_apex_surface_morph_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    surface_depth_mm: float,
    apex_fraction: float,
    confidence_min: float,
    max_component_volume_mm3: float,
    max_component_elongation: float,
    anchor_dilation_mm: float,
    protection_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = _zero_apex_surface_cleanup_metrics()
    liver = liver_mask.astype(bool, copy=False)
    if (
        not enabled
        or not liver.any()
        or max_component_volume_mm3 <= 0
        or max_component_elongation <= 0
        or (surface_depth_mm <= 0 and apex_fraction <= 0)
    ):
        return metrics

    depth = ndi.distance_transform_edt(liver, sampling=_sampling_zyx(spacing_xyz))
    surface_region = liver & (depth <= float(surface_depth_mm)) if surface_depth_mm > 0 else np.zeros(liver.shape, dtype=bool)
    apex_region = _liver_apex_mask(liver, apex_fraction=float(apex_fraction))
    target_region = surface_region | apex_region
    if not target_region.any():
        return metrics

    candidate = (
        (multilabel == PHASE_LABEL_IDS["venous"])
        & liver
        & target_region
        & (confidence < float(confidence_min))
    )
    if protection_mask is not None:
        candidate &= ~protection_mask.astype(bool, copy=False)
    if not candidate.any():
        return metrics

    effective_anchor_dilation_mm = max(
        float(anchor_dilation_mm) - 0.5 * min(float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])),
        0.0,
    )
    anchor_protection = _anchor_zone(
        anchor_mask,
        spacing_xyz=spacing_xyz,
        dilation_mm=effective_anchor_dilation_mm,
    )
    if anchor_protection.any():
        candidate &= ~anchor_protection
    if not candidate.any():
        return metrics

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(candidate, structure=structure)
    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 > float(max_component_volume_mm3):
            continue
        if _component_linearity(component) > float(max_component_elongation):
            continue
        surface_component = component & target_region[bbox]
        if not surface_component.any():
            continue
        if _component_elongation(component) >= float(max_component_elongation) * 0.85:
            prune[bbox] |= component
        else:
            prune[bbox] |= surface_component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["apex_surface_cleanup_voxels"] = int(prune.sum())
    metrics["apex_surface_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["apex_surface_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    apex_prune = prune & apex_region
    surface_prune = prune & surface_region & ~apex_region
    metrics["apex_surface_cleanup_by_region"]["apex"] = int(apex_prune.sum())
    metrics["apex_surface_cleanup_by_region"]["surface"] = int(surface_prune.sum())
    return metrics


def _zero_apex_subsurface_cleanup_metrics() -> dict[str, Any]:
    return {
        "apex_subsurface_cleanup_voxels": 0,
        "apex_subsurface_cleanup_components": 0,
        "apex_subsurface_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
        "apex_subsurface_cleanup_by_region": {
            "apex": 0,
            "subsurface": 0,
        },
        "apex_subsurface_cleanup_candidate_voxels": 0,
        "apex_subsurface_cleanup_protected_voxels": 0,
    }


def apply_apex_subsurface_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    apex_fraction: float,
    subsurface_min_depth_mm: float,
    subsurface_max_depth_mm: float,
    confidence_min: float,
    min_component_volume_mm3: float,
    max_component_volume_mm3: float,
    max_component_linearity: float,
    min_surface_fraction: float,
    anchor_dilation_mm: float,
    protection_mask: np.ndarray | None = None,
    min_surface_voxels: int | None = None,
) -> dict[str, Any]:
    metrics = _zero_apex_subsurface_cleanup_metrics()
    liver = liver_mask.astype(bool, copy=False)
    if (
        not enabled
        or not liver.any()
        or apex_fraction <= 0
        or subsurface_max_depth_mm <= 0
        or subsurface_min_depth_mm < 0
        or subsurface_max_depth_mm < subsurface_min_depth_mm
        or min_component_volume_mm3 < 0
        or max_component_volume_mm3 <= 0
        or max_component_linearity <= 0
        or min_surface_fraction < 0
    ):
        return metrics

    depth = ndi.distance_transform_edt(liver, sampling=_sampling_zyx(spacing_xyz))
    apex_region = _liver_apex_mask(liver, apex_fraction=float(apex_fraction))
    subsurface_region = apex_region & (depth >= float(subsurface_min_depth_mm)) & (depth <= float(subsurface_max_depth_mm))
    if not subsurface_region.any():
        return metrics

    eligible = (
        (multilabel == PHASE_LABEL_IDS["venous"])
        & liver
        & apex_region
        & (confidence <= float(confidence_min))
    )
    if protection_mask is not None:
        eligible &= ~protection_mask.astype(bool, copy=False)

    anchor_protection = _anchor_zone(
        anchor_mask,
        spacing_xyz=spacing_xyz,
        dilation_mm=anchor_dilation_mm,
    )
    if anchor_protection.any():
        eligible &= ~anchor_protection
    candidate = eligible & subsurface_region
    metrics["apex_subsurface_cleanup_candidate_voxels"] = int(candidate.sum())
    metrics["apex_subsurface_cleanup_protected_voxels"] = int((apex_region & subsurface_region & ~eligible).sum())
    if not candidate.any():
        return metrics

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    surface_voxel_min = int(min_surface_voxels) if min_surface_voxels is not None else int(np.ceil(float(min_component_volume_mm3) / max(voxel_volume, 1e-6)))
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(candidate, structure=structure)
    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    for bbox, component in _iter_labeled_component_views(labeled, count):
        subsurface_component = component & subsurface_region[bbox]
        if not subsurface_component.any():
            continue
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 < float(min_component_volume_mm3):
            continue
        if component_volume_mm3 > float(max_component_volume_mm3):
            continue
        if _component_linearity(component) > float(max_component_linearity):
            continue
        subsurface_fraction = float(subsurface_component.sum()) / max(int(component.sum()), 1)
        if subsurface_fraction < float(min_surface_fraction) and int(subsurface_component.sum()) < surface_voxel_min:
            continue
        prune[bbox] |= subsurface_component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["apex_subsurface_cleanup_voxels"] = int(prune.sum())
    metrics["apex_subsurface_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["apex_subsurface_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    apex_prune = prune & apex_region
    subsurface_prune = prune & subsurface_region
    metrics["apex_subsurface_cleanup_by_region"]["apex"] = int(apex_prune.sum())
    metrics["apex_subsurface_cleanup_by_region"]["subsurface"] = int(subsurface_prune.sum())
    return metrics


def _zero_outer_peripheral_cleanup_metrics() -> dict[str, Any]:
    return {
        "outer_peripheral_cleanup_voxels": 0,
        "outer_peripheral_cleanup_components": 0,
        "outer_peripheral_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }


def _zero_smv_portal_bridge_repair_metrics() -> dict[str, Any]:
    return {
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


def _zero_post_anchor_peripheral_audit_metrics() -> dict[str, Any]:
    return {
        "post_anchor_peripheral_cleanup_voxels": 0,
        "post_anchor_peripheral_cleanup_components": 0,
        "post_anchor_peripheral_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
    }


def _zero_liver_surface_sheet_cleanup_metrics() -> dict[str, Any]:
    return {
        "liver_surface_sheet_cleanup_voxels": 0,
        "liver_surface_sheet_cleanup_components": 0,
        "liver_surface_sheet_cleanup_by_label": {name: 0 for name in LABEL_NAMES.values()},
        "liver_surface_sheet_cleanup_protected_voxels": 0,
        "liver_surface_sheet_cleanup_candidate_voxels": 0,
    }


def apply_outer_peripheral_blob_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    body_mask: np.ndarray,
    liver_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    protection_mask: np.ndarray,
    anchor_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    max_component_volume_mm3: float,
    max_component_linearity: float,
    confidence_min: float,
    anchor_dilation_mm: float,
) -> dict[str, Any]:
    metrics = _zero_outer_peripheral_cleanup_metrics()
    if not enabled or max_component_volume_mm3 <= 0 or max_component_linearity <= 0:
        return metrics

    body = body_mask.astype(bool, copy=False)
    liver = liver_mask.astype(bool, copy=False)
    hard = hard_exclusion_mask.astype(bool, copy=False)
    protected = protection_mask.astype(bool, copy=False)
    candidate = (
        (multilabel > 0)
        & body
        & ~liver
        & ~hard
        & ~protected
        & (confidence < float(confidence_min))
    )
    if not candidate.any():
        return metrics

    anchor_protection = _anchor_zone(
        anchor_mask,
        spacing_xyz=spacing_xyz,
        dilation_mm=anchor_dilation_mm,
    )
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(candidate, structure=structure)
    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 > float(max_component_volume_mm3):
            continue
        if _component_linearity(component) > float(max_component_linearity):
            continue
        if anchor_protection.any() and (component & anchor_protection[bbox]).any():
            continue
        prune[bbox] |= component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["outer_peripheral_cleanup_voxels"] = int(prune.sum())
    metrics["outer_peripheral_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["outer_peripheral_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def _component_rows(mask: np.ndarray, *, spacing_xyz: tuple[float, float, float]) -> list[dict[str, Any]]:
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    labeled, count = ndi.label(mask, structure=structure)
    rows: list[dict[str, Any]] = []
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    for bbox, component in _iter_labeled_component_views(labeled, count):
        coords = np.argwhere(component)
        offset = np.array([axis.start for axis in bbox], dtype=np.int64)
        full_coords = coords + offset
        rows.append(
            {
                "id": int(labeled[bbox][component][0]),
                "bbox": bbox,
                "mask": component,
                "coords": full_coords,
                "volume_mm3": float(component.sum()) * voxel_volume,
            }
        )
    rows.sort(key=lambda row: int(row["coords"].shape[0]), reverse=True)
    return rows


def _zero_trunk_connectivity_metrics() -> dict[str, Any]:
    return {
        "intrahepatic_trunk_connected": True,
        "intrahepatic_trunk_components": 0,
        "intrahepatic_trunk_disconnected_components": 0,
        "intrahepatic_trunk_largest_disconnected_volume_mm3": 0.0,
        "intrahepatic_trunk_min_gap_mm": 0.0,
    }


def measure_intrahepatic_trunk_connectivity(
    multilabel: np.ndarray,
    *,
    trunk_seed_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    target_labels: tuple[str, ...] | list[str],
    min_component_volume_mm3: float,
) -> dict[str, Any]:
    metrics = _zero_trunk_connectivity_metrics()
    label_ids = _label_ids_from_names(target_labels)
    if not label_ids:
        return metrics

    liver = liver_mask.astype(bool, copy=False)
    target = np.isin(multilabel, list(label_ids)) & liver
    if not target.any():
        return metrics

    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(target, structure=structure)
    seed = trunk_seed_mask.astype(bool, copy=False) & target
    seed_labels = {int(value) for value in np.unique(labeled[seed]) if int(value) != 0}
    if not seed_labels:
        return metrics

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    rows: list[dict[str, Any]] = []
    trunk_component_coords: list[np.ndarray] = []
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_id = int(labeled[bbox][component][0])
        starts = np.array([axis.start for axis in bbox], dtype=np.int64)
        coords = np.argwhere(component) + starts
        volume_mm3 = float(component.sum()) * voxel_volume
        if component_id in seed_labels:
            trunk_component_coords.append(coords)
        if volume_mm3 >= float(min_component_volume_mm3):
            rows.append(
                {
                    "id": component_id,
                    "coords": coords,
                    "volume_mm3": volume_mm3,
                }
            )

    metrics["intrahepatic_trunk_components"] = len(rows)
    disconnected = [row for row in rows if int(row["id"]) not in seed_labels]
    metrics["intrahepatic_trunk_disconnected_components"] = len(disconnected)
    metrics["intrahepatic_trunk_connected"] = len(disconnected) == 0
    if not disconnected or not trunk_component_coords:
        return metrics

    metrics["intrahepatic_trunk_largest_disconnected_volume_mm3"] = max(
        float(row["volume_mm3"])
        for row in disconnected
    )
    trunk_coords = np.concatenate(trunk_component_coords, axis=0)
    min_gap = float("inf")
    for row in disconnected:
        gap_mm, _source, _target = _nearest_component_coords(
            row["coords"],
            trunk_coords,
            spacing_xyz=spacing_xyz,
        )
        min_gap = min(min_gap, float(gap_mm))
    metrics["intrahepatic_trunk_min_gap_mm"] = 0.0 if min_gap == float("inf") else float(min_gap)
    return metrics


def _nearest_component_voxels(
    source_coords: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    if source_coords.size == 0 or not target_mask.any():
        return float("inf"), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    indices = ndi.distance_transform_edt(
        ~target_mask.astype(bool, copy=False),
        sampling=_sampling_zyx(spacing_xyz),
        return_distances=False,
        return_indices=True,
    )
    nearest = indices[:, source_coords[:, 0], source_coords[:, 1], source_coords[:, 2]].T
    spacing_zyx = np.array(_sampling_zyx(spacing_xyz), dtype=np.float32)
    deltas = (source_coords - nearest).astype(np.float32) * spacing_zyx
    distances = np.sqrt((deltas * deltas).sum(axis=1))
    best = int(distances.argmin())
    return float(distances[best]), source_coords[best].astype(np.int64), nearest[best].astype(np.int64)


def _nearest_component_coords(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    if source_coords.size == 0 or target_coords.size == 0:
        return float("inf"), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    spacing_zyx = np.array(_sampling_zyx(spacing_xyz), dtype=np.float32)
    target_scaled = target_coords.astype(np.float32, copy=False) * spacing_zyx
    source_scaled = source_coords.astype(np.float32, copy=False) * spacing_zyx
    tree = cKDTree(target_scaled)
    distances, nearest_indices = tree.query(source_scaled, k=1)
    min_distance = float(np.min(distances))
    ties = np.flatnonzero(distances <= (min_distance + 1e-6))
    if ties.size > 1:
        source_centroid = source_scaled.mean(axis=0)
        target_centroid = target_scaled.mean(axis=0)
        tie_target_scaled = target_scaled[nearest_indices[ties].astype(np.int64)]
        tie_source_scaled = source_scaled[ties]
        scores = np.linalg.norm(tie_source_scaled - source_centroid, axis=1) + np.linalg.norm(tie_target_scaled - target_centroid, axis=1)
        best = int(ties[int(np.argmin(scores))])
    else:
        best = int(ties[0])
    return (
        float(distances[best]),
        source_coords[best].astype(np.int64),
        target_coords[int(nearest_indices[best])].astype(np.int64),
    )


def _coords_mask_in_bbox(coords: np.ndarray, bbox: tuple[slice, ...]) -> np.ndarray:
    starts = np.array([axis.start for axis in bbox], dtype=np.int64)
    stops = np.array([axis.stop for axis in bbox], dtype=np.int64)
    inside = np.all((coords >= starts) & (coords < stops), axis=1)
    local = coords[inside] - starts
    shape = tuple(int(axis.stop - axis.start) for axis in bbox)
    mask = np.zeros(shape, dtype=bool)
    if local.size:
        mask[tuple(local.T)] = True
    return mask


def _line_mask(shape: tuple[int, ...], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    steps = int(np.max(np.abs(end - start))) + 1
    coords = np.rint(np.linspace(start, end, max(steps, 2))).astype(np.int64)
    coords = np.clip(coords, 0, np.array(shape, dtype=np.int64) - 1)
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    return mask


def _binary_close_mm(mask: np.ndarray, *, spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0 or not mask.any():
        return mask.astype(bool, copy=True)
    closed = _dilate_mm(mask, spacing_xyz=spacing_xyz, distance_mm=radius_mm)
    erode_distance = ndi.distance_transform_edt(closed, sampling=_sampling_zyx(spacing_xyz))
    return erode_distance > float(radius_mm)


def _surface_band_by_iterations(mask: np.ndarray, *, spacing_xyz: tuple[float, float, float], depth_mm: float) -> np.ndarray:
    source = mask.astype(bool, copy=False)
    if depth_mm <= 0 or not source.any():
        return np.zeros(source.shape, dtype=bool)
    surface = np.zeros(source.shape, dtype=bool)
    spacing_zyx = _sampling_zyx(spacing_xyz)
    for axis, spacing in enumerate(spacing_zyx):
        steps = max(1, int(np.ceil(float(depth_mm) / max(float(spacing), 1e-6))))
        for offset in range(1, steps + 1):
            distance_mm = offset * float(spacing)
            if distance_mm > float(depth_mm) + 1e-6:
                break
            before_src = [slice(None)] * source.ndim
            before_neighbor = [slice(None)] * source.ndim
            before_src[axis] = slice(offset, None)
            before_neighbor[axis] = slice(None, -offset)
            src_tuple = tuple(before_src)
            neighbor_tuple = tuple(before_neighbor)
            surface[src_tuple] |= source[src_tuple] & ~source[neighbor_tuple]

            after_src = [slice(None)] * source.ndim
            after_neighbor = [slice(None)] * source.ndim
            after_src[axis] = slice(None, -offset)
            after_neighbor[axis] = slice(offset, None)
            src_tuple = tuple(after_src)
            neighbor_tuple = tuple(after_neighbor)
            surface[src_tuple] |= source[src_tuple] & ~source[neighbor_tuple]
    return surface & source


def _dilate_mm(mask: np.ndarray, *, spacing_xyz: tuple[float, float, float], distance_mm: float) -> np.ndarray:
    if distance_mm <= 0 or not mask.any():
        return mask.astype(bool, copy=True)
    distance = ndi.distance_transform_edt(~mask.astype(bool, copy=False), sampling=_sampling_zyx(spacing_xyz))
    return distance <= float(distance_mm)


def apply_smv_portal_bridge_repair(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    protection_mask: np.ndarray,
    portal_hu: np.ndarray,
    venous_hu: np.ndarray,
    portal_coverage: np.ndarray,
    venous_coverage: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    max_gap_mm: float,
    corridor_radius_mm: float,
    endpoint_min_volume_mm3: float,
    min_portal_minus_venous_hu: float,
    fallback_centerline_enabled: bool,
    bridge_confidence: float,
    morphological_tube_fill_enabled: bool = False,
    tube_radius_mm: float = 0.0,
    closing_radius_mm: float = 0.0,
    min_evidence_fraction: float = 0.0,
    max_fill_to_evidence_ratio: float = float("inf"),
) -> dict[str, Any]:
    metrics = _zero_smv_portal_bridge_repair_metrics()
    if not enabled or max_gap_mm <= 0 or corridor_radius_mm <= 0:
        return metrics

    portal = multilabel == PHASE_LABEL_IDS["portal"]
    rows = [
        row
        for row in _component_rows(portal, spacing_xyz=spacing_xyz)
        if float(row["volume_mm3"]) >= float(endpoint_min_volume_mm3)
    ]
    rows = rows[:16]
    if len(rows) < 2:
        return metrics

    allowed = (
        body_mask.astype(bool, copy=False)
        & ~hard_exclusion_mask.astype(bool, copy=False)
        & protection_mask.astype(bool, copy=False)
    )
    if not allowed.any():
        return metrics

    repaired = np.zeros(multilabel.shape, dtype=bool)
    fallback_repaired = np.zeros(multilabel.shape, dtype=bool)
    evidence_repaired = np.zeros(multilabel.shape, dtype=bool)
    morph_fill_repaired = np.zeros(multilabel.shape, dtype=bool)
    repaired_pairs = 0
    max_repaired_gap = 0.0

    valid_phase = portal_coverage.astype(bool, copy=False) & venous_coverage.astype(bool, copy=False)
    hu_delta = portal_hu.astype(np.float32, copy=False) - venous_hu.astype(np.float32, copy=False)
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            gap_mm, source, target = _nearest_component_coords(
                right["coords"],
                left["coords"],
                spacing_xyz=spacing_xyz,
            )
            if gap_mm > float(max_gap_mm):
                continue

            tube_radius = float(tube_radius_mm) if tube_radius_mm > 0 else float(corridor_radius_mm)
            bbox_radius = max(float(corridor_radius_mm), tube_radius, float(closing_radius_mm)) + 2.0
            mins = np.maximum(np.minimum(source, target) - int(np.ceil(bbox_radius)) - 2, 0)
            maxs = np.minimum(np.maximum(source, target) + int(np.ceil(bbox_radius)) + 3, np.array(multilabel.shape))
            bbox = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(multilabel.ndim))
            starts = np.array([axis.start for axis in bbox], dtype=np.int64)
            local_shape = tuple(int(axis.stop - axis.start) for axis in bbox)
            local_source = (source - starts).astype(np.int64)
            local_target = (target - starts).astype(np.int64)
            line = _line_mask(local_shape, local_source, local_target)
            allowed_local = allowed[bbox]
            portal_local = portal[bbox]
            multilabel_local = multilabel[bbox]
            valid_phase_local = valid_phase[bbox]
            hu_delta_local = hu_delta[bbox]

            corridor = _dilate_mm(line, spacing_xyz=spacing_xyz, distance_mm=float(corridor_radius_mm)) & allowed_local
            tube = _dilate_mm(line, spacing_xyz=spacing_xyz, distance_mm=tube_radius) & allowed_local
            search_region = tube if morphological_tube_fill_enabled else corridor
            candidate = (
                search_region
                & ~portal_local
                & (
                    (multilabel_local == PHASE_LABEL_IDS["venous"])
                    | ((multilabel_local == 0) & valid_phase_local & (hu_delta_local >= float(min_portal_minus_venous_hu)))
                )
            )
            evidence = candidate.copy()
            if morphological_tube_fill_enabled:
                tube_candidate_voxels = int((search_region & ~portal_local).sum())
                evidence_voxels = int(evidence.sum())
                evidence_fraction = evidence_voxels / max(tube_candidate_voxels, 1)
                if evidence_voxels == 0 or evidence_fraction < float(min_evidence_fraction):
                    metrics["smv_portal_bridge_repair_rejected_pairs"] += 1
                    metrics["smv_portal_bridge_repair_rejected_by_reason"]["insufficient_evidence"] += 1
                    continue
                closed = _binary_close_mm(evidence, spacing_xyz=spacing_xyz, radius_mm=float(closing_radius_mm))
                candidate = (evidence | (closed & search_region & ~portal_local)) & search_region & ~portal_local
                fill_ratio = int(candidate.sum()) / max(evidence_voxels, 1)
                if fill_ratio > float(max_fill_to_evidence_ratio):
                    metrics["smv_portal_bridge_repair_rejected_pairs"] += 1
                    metrics["smv_portal_bridge_repair_rejected_by_reason"]["excessive_fill"] += 1
                    continue
            elif fallback_centerline_enabled and not (line & ~allowed_local).any():
                line_candidate = line & allowed_local & ~portal_local
                if line_candidate.any():
                    fallback_repaired[bbox] |= line_candidate & ~candidate
                    candidate |= line_candidate
            if not candidate.any():
                continue

            linked = (
                candidate
                | _coords_mask_in_bbox(left["coords"], bbox)
                | _coords_mask_in_bbox(right["coords"], bbox)
            )
            labeled, _ = ndi.label(linked, structure=structure)
            left_label = int(labeled[tuple(local_target)])
            right_label = int(labeled[tuple(local_source)])
            if left_label == 0 or left_label != right_label:
                if morphological_tube_fill_enabled:
                    metrics["smv_portal_bridge_repair_rejected_pairs"] += 1
                    metrics["smv_portal_bridge_repair_rejected_by_reason"]["not_connected"] += 1
                continue
            new_candidate = candidate & ~repaired[bbox]
            if not new_candidate.any():
                continue
            repaired[bbox] |= new_candidate
            evidence_repaired[bbox] |= evidence & new_candidate
            morph_fill_repaired[bbox] |= new_candidate & (multilabel_local == 0)
            repaired_pairs += 1
            max_repaired_gap = max(max_repaired_gap, float(gap_mm))

    if not repaired.any():
        return metrics

    changed = repaired & (multilabel != PHASE_LABEL_IDS["portal"])
    multilabel[changed] = PHASE_LABEL_IDS["portal"]
    confidence[changed] = np.maximum(confidence[changed], float(bridge_confidence))
    metrics["smv_portal_bridge_repair_voxels"] = int(changed.sum())
    metrics["smv_portal_bridge_repair_pairs"] = int(repaired_pairs)
    metrics["smv_portal_bridge_repair_fallback_voxels"] = int((changed & fallback_repaired).sum())
    metrics["smv_portal_bridge_repair_max_gap_mm"] = float(max_repaired_gap)
    metrics["smv_portal_bridge_repair_evidence_voxels"] = int((changed & evidence_repaired).sum())
    metrics["smv_portal_bridge_repair_morph_fill_voxels"] = int((changed & morph_fill_repaired).sum())
    return metrics


def apply_post_anchor_peripheral_component_audit(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    envelope_seed_mask: np.ndarray,
    core_anchor_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    organ_envelope_dilation_mm: float,
    core_anchor_protection_mm: float,
    min_component_volume_mm3: float,
    max_component_linearity: float,
    confidence_max: float,
) -> dict[str, Any]:
    metrics = _zero_post_anchor_peripheral_audit_metrics()
    if (
        not enabled
        or min_component_volume_mm3 <= 0
        or max_component_linearity <= 0
        or organ_envelope_dilation_mm < 0
        or core_anchor_protection_mm < 0
    ):
        return metrics

    envelope = _dilate_mm(
        envelope_seed_mask.astype(bool, copy=False),
        spacing_xyz=spacing_xyz,
        distance_mm=float(organ_envelope_dilation_mm),
    )
    core_protection = _dilate_mm(
        core_anchor_mask.astype(bool, copy=False),
        spacing_xyz=spacing_xyz,
        distance_mm=float(core_anchor_protection_mm),
    )
    candidate = (
        (multilabel > 0)
        & body_mask.astype(bool, copy=False)
        & ~hard_exclusion_mask.astype(bool, copy=False)
        & ~envelope
        & ~core_protection
        & (confidence <= float(confidence_max))
    )
    if not candidate.any():
        return metrics

    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(candidate, structure=structure)
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 < float(min_component_volume_mm3):
            continue
        if _component_linearity(component) > float(max_component_linearity):
            continue
        prune[bbox] |= component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["post_anchor_peripheral_cleanup_voxels"] = int(prune.sum())
    metrics["post_anchor_peripheral_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["post_anchor_peripheral_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def apply_liver_surface_sheet_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    liver_mask: np.ndarray,
    body_mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    core_anchor_mask: np.ndarray,
    bridge_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    surface_depth_mm: float,
    min_component_volume_mm3: float,
    max_component_volume_mm3: float,
    max_component_linearity: float,
    min_surface_fraction: float,
    confidence_max: float,
    core_anchor_protection_mm: float,
    bridge_protection_mm: float,
    target_labels: tuple[str, ...] | list[str],
    min_surface_voxels: int | None = None,
) -> dict[str, Any]:
    metrics = _zero_liver_surface_sheet_cleanup_metrics()
    liver = liver_mask.astype(bool, copy=False)
    if (
        not enabled
        or not liver.any()
        or surface_depth_mm <= 0
        or min_component_volume_mm3 < 0
        or max_component_volume_mm3 <= 0
        or max_component_linearity <= 0
        or min_surface_fraction < 0
    ):
        return metrics

    label_ids = {
        PHASE_LABEL_IDS[name]
        for name in target_labels
        if isinstance(name, str) and name in PHASE_LABEL_IDS
    }
    if not label_ids:
        return metrics

    surface_band = _surface_band_by_iterations(
        liver,
        spacing_xyz=spacing_xyz,
        depth_mm=float(surface_depth_mm),
    )
    core_protection = _dilate_mm(
        core_anchor_mask.astype(bool, copy=False),
        spacing_xyz=spacing_xyz,
        distance_mm=float(core_anchor_protection_mm),
    )
    bridge_protection = _dilate_mm(
        bridge_mask.astype(bool, copy=False),
        spacing_xyz=spacing_xyz,
        distance_mm=float(bridge_protection_mm),
    )
    protected = core_protection | bridge_protection
    target = np.isin(multilabel, list(label_ids))
    eligible = (
        target
        & liver
        & body_mask.astype(bool, copy=False)
        & ~hard_exclusion_mask.astype(bool, copy=False)
        & ~protected
        & (confidence <= float(confidence_max))
    )
    candidate = eligible & surface_band
    metrics["liver_surface_sheet_cleanup_candidate_voxels"] = int(candidate.sum())
    metrics["liver_surface_sheet_cleanup_protected_voxels"] = int((target & surface_band & protected).sum())
    if not candidate.any() or not eligible.any():
        return metrics

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    surface_voxel_min = int(min_surface_voxels) if min_surface_voxels is not None else int(np.ceil(float(min_component_volume_mm3) / max(voxel_volume, 1e-6)))
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(eligible, structure=structure)
    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    for bbox, component in _iter_labeled_component_views(labeled, count):
        surface_component = component & surface_band[bbox]
        if not surface_component.any():
            continue
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 < float(min_component_volume_mm3):
            continue
        if component_volume_mm3 > float(max_component_volume_mm3):
            continue
        if _component_linearity(component) > float(max_component_linearity):
            continue
        surface_fraction = float(surface_component.sum()) / max(int(component.sum()), 1)
        if surface_fraction < float(min_surface_fraction) and int(surface_component.sum()) < surface_voxel_min:
            continue
        if _component_elongation(component) >= float(max_component_linearity) * 0.85:
            prune[bbox] |= component
        else:
            prune[bbox] |= surface_component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["liver_surface_sheet_cleanup_voxels"] = int(prune.sum())
    metrics["liver_surface_sheet_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["liver_surface_sheet_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics


def apply_isolated_liver_blob_cleanup(
    multilabel: np.ndarray,
    confidence: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    enabled: bool,
    max_component_volume_mm3: float,
    max_component_elongation: float,
    confidence_min: float,
    anchor_dilation_mm: float,
    protection_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = _zero_isolated_blob_cleanup_metrics()
    liver = liver_mask.astype(bool, copy=False)
    if not enabled or max_component_volume_mm3 <= 0 or not liver.any():
        return metrics

    candidate = (
        (multilabel == PHASE_LABEL_IDS["venous"])
        & liver
        & (confidence < float(confidence_min))
    )
    if protection_mask is not None:
        candidate &= ~protection_mask.astype(bool, copy=False)
    if not candidate.any():
        return metrics

    anchors = anchor_mask.astype(bool, copy=False)
    if anchors.any() and anchor_dilation_mm > 0:
        sampling_zyx = (
            float(spacing_xyz[2]),
            float(spacing_xyz[1]),
            float(spacing_xyz[0]),
        )
        anchor_distance = ndi.distance_transform_edt(~anchors, sampling=sampling_zyx)
        anchor_zone = anchor_distance <= float(anchor_dilation_mm)
    else:
        anchor_zone = anchors

    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    structure = ndi.generate_binary_structure(multilabel.ndim, 1)
    labeled, count = ndi.label(candidate, structure=structure)
    prune = np.zeros(multilabel.shape, dtype=bool)
    removed_components = 0
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_volume_mm3 = float(component.sum()) * voxel_volume
        if component_volume_mm3 > float(max_component_volume_mm3):
            continue
        if _component_elongation(component) > float(max_component_elongation):
            continue
        if anchor_zone.any() and (component & anchor_zone[bbox]).any():
            continue
        prune[bbox] |= component
        removed_components += 1

    if not prune.any():
        return metrics

    labels_before = multilabel[prune].copy()
    multilabel[prune] = 0
    confidence[prune] = 0.0
    metrics["isolated_blob_cleanup_voxels"] = int(prune.sum())
    metrics["isolated_blob_cleanup_components"] = int(removed_components)
    for label_id, name in LABEL_NAMES.items():
        metrics["isolated_blob_cleanup_by_label"][name] = int((labels_before == label_id).sum())
    return metrics
