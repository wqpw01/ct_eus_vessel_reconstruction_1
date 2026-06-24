# CT-0021 v13 Trunk Continuity And Apex Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v12b Git 基线之上，解决两个明确遗留问题：肝内主血管与主干断裂，以及肝尖浅表下约 1cm 片状/破洞网状组织未清理。

**Architecture:** 先补上组件级诊断指标，避免再只看 `smv_portal_bridge_repair_voxels`。然后新增肝内主血管到主干的证据约束 reconnect，并把 reconnect 后的 trunk 组件作为后续清理保护区。最后扩大肝尖浅表下清理窗口到真实病例能覆盖的 apex band，并用 trunk protection 防止清理把主血管再切断。

**Tech Stack:** Python 3.12, NumPy, SciPy ndimage/cKDTree, SimpleITK, pytest, YAML config.

---

## 明确问题记录

- **问题 A：肝内主血管与主干断裂。**
  - v12b 中 `smv_portal_bridge_repair_voxels: 1854`，说明 SMV/portal 局部 bridge 有补全。
  - 但用户 Slicer 复核显示肝内主血管没有和主干连接上，说明现有指标没有覆盖“肝内大分支与主干是否属于同一 3D 连通组件”。
  - 改进方向：新增 trunk connectivity audit，再新增只在有相位/HU/候选证据的局部 corridor 内补连的 `intrahepatic_trunk_reconnect`。

- **问题 B：肝尖浅表下约 1cm 片状/破洞网状组织未清理。**
  - v12b 中 `apex_subsurface_cleanup_voxels: 0`，说明现有清理没有打到目标区域。
  - 既有 v11 参数 `apex_fraction: 0.12` 只覆盖最尖端区域；真实病例扫参显示 `apex_fraction: 0.20`、`subsurface_min_depth_mm: 2`、`subsurface_max_depth_mm: 8` 才能产生候选。
  - 改进方向：v13 用更大的 apex window 和浅表下深度窗口清理 venous 片状组织，同时使用 reconnect 后的 protected trunk 作为保护区。

## 文件结构

- Modify: `src/ct_eus_vessel/postprocess.py`
  - 新增 `measure_intrahepatic_trunk_connectivity`
  - 新增 `apply_intrahepatic_trunk_reconnect`
  - 扩展 `apply_apex_subsurface_cleanup` 的可配置项：允许传入更完整的 `protection_mask`，并记录更多候选/拒绝原因。

- Modify: `src/ct_eus_vessel/pipeline.py`
  - 在 SMV/portal bridge repair 后、任何肝尖/肝表面清理前执行 trunk connectivity audit。
  - 执行 `apply_intrahepatic_trunk_reconnect`。
  - 用 reconnect 后 trunk seed 重新构建 `protected_trunk_mask`。
  - 将 `apex_surface_cleanup` 和 `apex_subsurface_cleanup` 的 protection 从 `bridge_seed_mask` 升级为 `bridge_seed_mask | protected_trunk_mask`。
  - 输出新增质量指标。

- Create: `config/ct0021_v13.yaml`
  - 基于 v10/v12b 稳定 SMV 设置。
  - 开启 `intrahepatic_trunk_reconnect`。
  - 开启参数修正后的 `apex_subsurface_cleanup`。

- Modify: `tests/test_postprocess.py`
  - 增加 trunk connectivity audit 纯函数测试。
  - 增加 trunk reconnect 有证据补连测试。
  - 增加 trunk reconnect 无证据拒绝测试。
  - 增加 apex subsurface 扩大窗口但保留 protected trunk 的测试。

- Modify: `tests/test_pipeline_auto.py`
  - 增加 v13 pipeline 集成测试：SMV bridge 保持、肝内主血管接回主干、肝尖浅表下片状组织被删除。

- Modify: `tests/test_config.py`
  - 增加 v13 配置回归测试。

- Modify: `docs/ct_eus_vessel_reconstruction_iterations.md`
  - 追加 v13 设计、真实 CT 运行结果和 Slicer 复核要点。

---

### Task 1: Add v13 Config Test First

**Files:**
- Create: `config/ct0021_v13.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Add to `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py::test_ct0021_v13_config_connects_trunk_and_expands_apex_subsurface_cleanup
```

Expected: FAIL because `config/ct0021_v13.yaml` does not exist.

- [ ] **Step 3: Create `config/ct0021_v13.yaml`**

Start from `config/ct0021_v10.yaml` and add:

```yaml
vessel_extraction:
  hilar_protection:
    enabled: true
    distance_mm: 12
  smv_portal_bridge_protection:
    enabled: true
    distance_mm: 22
  final_liver_surface_cleanup:
    enabled: true
    surface_depth_mm: 8
    confidence_min: 0.78
  portal_from_venous_relabel:
    enabled: true
    min_portal_minus_venous_hu: 30
    protected_min_portal_minus_venous_hu: 20
  deep_liver_cleanup:
    enabled: false
  isolated_liver_blob_cleanup:
    enabled: true
    max_component_volume_mm3: 48
    max_component_elongation: 2.0
    confidence_min: 0.80
    anchor_dilation_mm: 4
  apex_surface_morph_cleanup:
    enabled: true
    surface_depth_mm: 10
    apex_fraction: 0.12
    confidence_min: 0.82
    max_component_volume_mm3: 96
    max_component_elongation: 2.4
    anchor_dilation_mm: 4
  apex_subsurface_cleanup:
    enabled: true
    apex_fraction: 0.20
    subsurface_min_depth_mm: 2
    subsurface_max_depth_mm: 8
    confidence_min: 0.80
    min_component_volume_mm3: 120
    max_component_volume_mm3: 1800
    max_component_linearity: 4.5
    min_surface_fraction: 0.30
    anchor_dilation_mm: 2
    protection_source: protected_trunk
  intrahepatic_trunk_reconnect:
    enabled: true
    target_labels:
      - portal
      - venous
    max_gap_mm: 18
    corridor_radius_mm: 2.5
    tube_radius_mm: 2.5
    closing_radius_mm: 1.2
    min_component_volume_mm3: 300
    min_evidence_fraction: 0.25
    max_fill_to_evidence_ratio: 2.0
    bridge_confidence: 0.86
  outer_peripheral_blob_cleanup:
    enabled: false
    max_component_volume_mm3: 160
    max_component_linearity: 2.2
    confidence_min: 0.82
    anchor_dilation_mm: 6
  smv_portal_bridge_repair:
    enabled: true
    max_gap_mm: 30
    corridor_radius_mm: 5
    endpoint_min_volume_mm3: 300
    min_portal_minus_venous_hu: 10
    fallback_centerline_enabled: false
    bridge_confidence: 0.85
    morphological_tube_fill_enabled: true
    tube_radius_mm: 3.0
    closing_radius_mm: 1.5
    min_evidence_fraction: 0.35
    max_fill_to_evidence_ratio: 1.75
  post_anchor_peripheral_component_audit:
    enabled: true
    organ_envelope_dilation_mm: 18
    core_anchor_protection_mm: 6
    min_component_volume_mm3: 96
    max_component_linearity: 3.0
    confidence_max: 1.01
    organ_envelope_masks:
      - liver
      - pancreas
      - spleen
      - stomach
      - duodenum
      - aorta
      - inferior_vena_cava
      - portal_vein_and_splenic_vein
  liver_surface_sheet_cleanup:
    enabled: true
    surface_depth_mm: 15
    min_component_volume_mm3: 200
    max_component_volume_mm3: 12000
    max_component_linearity: 4.5
    min_surface_fraction: 0.55
    confidence_max: 1.01
    core_anchor_protection_mm: 8
    bridge_protection_mm: 5
    target_labels:
      - arterial
      - portal
      - venous
```

- [ ] **Step 4: Run config tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/ct0021_v13.yaml tests/test_config.py
git commit -m "test: add v13 trunk and apex config baseline"
```

---

### Task 2: Add Trunk Connectivity Audit

**Files:**
- Modify: `src/ct_eus_vessel/postprocess.py`
- Modify: `tests/test_postprocess.py`

- [ ] **Step 1: Write failing connectivity audit test**

Add to `tests/test_postprocess.py`:

```python
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
```

- [ ] **Step 2: Run failing test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py::test_measure_intrahepatic_trunk_connectivity_reports_disconnected_large_branch
```

Expected: FAIL because `measure_intrahepatic_trunk_connectivity` does not exist.

- [ ] **Step 3: Implement audit helper**

Add to `src/ct_eus_vessel/postprocess.py` near component helpers:

```python
def _label_ids_from_names(names: tuple[str, ...] | list[str]) -> set[int]:
    return {
        PHASE_LABEL_IDS[name]
        for name in names
        if isinstance(name, str) and name in PHASE_LABEL_IDS
    }


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
    seed = trunk_seed_mask.astype(bool, copy=False) & target
    if not target.any() or not seed.any():
        return metrics

    labeled, count = ndi.label(target, structure=ndi.generate_binary_structure(multilabel.ndim, 1))
    seed_labels = set(int(value) for value in np.unique(labeled[seed]) if int(value) != 0)
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    rows = []
    for bbox, component in _iter_labeled_component_views(labeled, count):
        component_id = int(labeled[bbox][component][0])
        volume_mm3 = float(component.sum()) * voxel_volume
        if volume_mm3 >= float(min_component_volume_mm3):
            rows.append((component_id, bbox, component, volume_mm3))

    metrics["intrahepatic_trunk_components"] = len(rows)
    disconnected = [row for row in rows if row[0] not in seed_labels]
    metrics["intrahepatic_trunk_disconnected_components"] = len(disconnected)
    metrics["intrahepatic_trunk_connected"] = len(disconnected) == 0
    if not disconnected:
        return metrics

    metrics["intrahepatic_trunk_largest_disconnected_volume_mm3"] = max(row[3] for row in disconnected)
    seed_coords = np.argwhere(seed)
    min_gap = float("inf")
    for _component_id, bbox, component, _volume in disconnected:
        starts = np.array([axis.start for axis in bbox], dtype=np.int64)
        coords = np.argwhere(component) + starts
        gap_mm, _source, _target = _nearest_component_coords(coords, seed_coords, spacing_xyz=spacing_xyz)
        min_gap = min(min_gap, float(gap_mm))
    metrics["intrahepatic_trunk_min_gap_mm"] = 0.0 if min_gap == float("inf") else float(min_gap)
    return metrics
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py::test_measure_intrahepatic_trunk_connectivity_reports_disconnected_large_branch
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ct_eus_vessel/postprocess.py tests/test_postprocess.py
git commit -m "feat: audit intrahepatic trunk connectivity"
```

---

### Task 3: Add Evidence-Constrained Intrahepatic Trunk Reconnect

**Files:**
- Modify: `src/ct_eus_vessel/postprocess.py`
- Modify: `tests/test_postprocess.py`

- [ ] **Step 1: Write reconnect positive test**

Add to `tests/test_postprocess.py`:

```python
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
```

- [ ] **Step 2: Write reconnect rejection test**

Add to `tests/test_postprocess.py`:

```python
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
```

- [ ] **Step 3: Run failing reconnect tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py -k "intrahepatic_trunk_reconnect"
```

Expected: FAIL because `apply_intrahepatic_trunk_reconnect` does not exist.

- [ ] **Step 4: Implement reconnect**

Implementation rules:

- Only connect large disconnected liver components whose nearest gap to trunk is `<= max_gap_mm`.
- Allowed region is `liver & body & ~hard_exclusion`.
- Evidence is `candidate_mask` inside a tube around the nearest component-to-trunk line.
- Fill may include binary closing inside the local tube, but reject if evidence fraction is too low or fill/evidence ratio is too high.
- Use the disconnected component's majority label as the filled label, so a venous branch stays venous and portal stays portal.

Add metric initializer:

```python
def _zero_trunk_reconnect_metrics() -> dict[str, Any]:
    return {
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
```

Then implement `apply_intrahepatic_trunk_reconnect` using the same local-bbox pattern as `apply_smv_portal_bridge_repair`, reusing `_nearest_component_coords`, `_line_mask`, `_dilate_mm`, `_binary_close_mm`, and `_coords_mask_in_bbox`.

- [ ] **Step 5: Run postprocess reconnect tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py -k "intrahepatic_trunk_connectivity or intrahepatic_trunk_reconnect"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ct_eus_vessel/postprocess.py tests/test_postprocess.py
git commit -m "feat: reconnect intrahepatic trunk branches with evidence"
```

---

### Task 4: Protect Reconnected Trunk Before Apex Cleanup

**Files:**
- Modify: `src/ct_eus_vessel/pipeline.py`
- Modify: `tests/test_pipeline_auto.py`

- [ ] **Step 1: Write failing pipeline order test**

Add a v13 integration test modeled after `test_run_pipeline_v11_preserves_bridge_and_cleans_apex_subsurface_sheet`:

```python
def test_run_pipeline_v13_reconnects_trunk_before_apex_cleanup_and_keeps_smv_bridge(tmp_path: Path, monkeypatch) -> None:
    # Build synthetic portal/venous phases with:
    # 1. two portal components that require SMV bridge repair,
    # 2. one large venous intrahepatic branch separated from trunk by an evidence corridor,
    # 3. one apex subsurface venous sheet not connected to trunk.
    # Assert:
    # - SMV bridge voxels are portal label 2.
    # - reconnect corridor voxels are venous label 3.
    # - apex sheet voxels are removed.
    # - trunk connectivity after repair is true.
    # Use config/ct0021_v13.yaml or a local minimal YAML with the same v13 keys.
```

The test body should follow existing fixtures in `tests/test_pipeline_auto.py`: monkeypatch `index_dicom_series`, `_candidate_images_by_uid`, `_phase_candidate`, and `ensure_totalseg_multilabel`.

- [ ] **Step 2: Run failing pipeline test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py::test_run_pipeline_v13_reconnects_trunk_before_apex_cleanup_and_keeps_smv_bridge
```

Expected: FAIL because pipeline does not call the new reconnect function or expose new metrics.

- [ ] **Step 3: Wire reconnect into pipeline**

Modify `src/ct_eus_vessel/pipeline.py`:

```python
intrahepatic_trunk_connectivity_before = {
    "intrahepatic_trunk_connected": True,
    "intrahepatic_trunk_components": 0,
    "intrahepatic_trunk_disconnected_components": 0,
    "intrahepatic_trunk_largest_disconnected_volume_mm3": 0.0,
    "intrahepatic_trunk_min_gap_mm": 0.0,
}
intrahepatic_trunk_connectivity_after = dict(intrahepatic_trunk_connectivity_before)
intrahepatic_trunk_reconnect = _zero_trunk_reconnect_metrics()
```

After `apply_smv_portal_bridge_repair` and before `protected_trunk_mask` is computed:

```python
trunk_seed_mask = (
    bridge_repaired_mask
    | portal_relabel_bridge_mask
    | hilar_protection_mask
    | smv_portal_protection_mask
)
trunk_reconnect_cfg = vessel_cfg.get("intrahepatic_trunk_reconnect", {})
if (
    not manual_label
    and liver_mask is not None
    and isinstance(trunk_reconnect_cfg, dict)
    and bool(trunk_reconnect_cfg.get("enabled", False))
):
    target_labels = trunk_reconnect_cfg.get("target_labels", ["portal", "venous"])
    intrahepatic_trunk_connectivity_before = measure_intrahepatic_trunk_connectivity(
        fused.multilabel,
        trunk_seed_mask=trunk_seed_mask,
        liver_mask=liver_mask,
        spacing_xyz=_spacing_xyz(reference),
        target_labels=target_labels,
        min_component_volume_mm3=float(trunk_reconnect_cfg.get("min_component_volume_mm3", 0.0)),
    )
    candidate_mask = portal_mask | venous_mask | intrahepatic_recovery_mask
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
        target_labels=target_labels,
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
        target_labels=target_labels,
        min_component_volume_mm3=float(trunk_reconnect_cfg.get("min_component_volume_mm3", 0.0)),
    )
```

Then compute:

```python
protected_trunk_mask = keep_components_near_anchors(
    fused.multilabel > 0,
    trunk_seed_mask,
    dilation_voxels=4,
)
```

And set:

```python
apex_protection_mask = (bridge_seed_mask | protected_trunk_mask) if (bridge_seed_mask.any() or protected_trunk_mask.any()) else None
apex_subsurface_protection_mask = apex_protection_mask
```

- [ ] **Step 4: Add metrics to `quality_metrics`**

Add:

```python
"intrahepatic_trunk_connected_before": bool(intrahepatic_trunk_connectivity_before["intrahepatic_trunk_connected"]),
"intrahepatic_trunk_connected_after": bool(intrahepatic_trunk_connectivity_after["intrahepatic_trunk_connected"]),
"intrahepatic_trunk_disconnected_components_before": int(intrahepatic_trunk_connectivity_before["intrahepatic_trunk_disconnected_components"]),
"intrahepatic_trunk_disconnected_components_after": int(intrahepatic_trunk_connectivity_after["intrahepatic_trunk_disconnected_components"]),
"intrahepatic_trunk_min_gap_mm_before": float(intrahepatic_trunk_connectivity_before["intrahepatic_trunk_min_gap_mm"]),
"intrahepatic_trunk_min_gap_mm_after": float(intrahepatic_trunk_connectivity_after["intrahepatic_trunk_min_gap_mm"]),
"intrahepatic_trunk_reconnect_voxels": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_voxels"]),
"intrahepatic_trunk_reconnect_pairs": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_pairs"]),
"intrahepatic_trunk_reconnect_max_gap_mm": float(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_max_gap_mm"]),
"intrahepatic_trunk_reconnect_evidence_voxels": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_evidence_voxels"]),
"intrahepatic_trunk_reconnect_morph_fill_voxels": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_morph_fill_voxels"]),
"intrahepatic_trunk_reconnect_rejected_pairs": int(intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_rejected_pairs"]),
"intrahepatic_trunk_reconnect_rejected_by_reason": intrahepatic_trunk_reconnect["intrahepatic_trunk_reconnect_rejected_by_reason"],
```

- [ ] **Step 5: Run pipeline v13 test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py::test_run_pipeline_v13_reconnects_trunk_before_apex_cleanup_and_keeps_smv_bridge
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ct_eus_vessel/pipeline.py tests/test_pipeline_auto.py
git commit -m "feat: reconnect trunk before apex cleanup"
```

---

### Task 5: Verify Apex Subsurface Cleanup Window

**Files:**
- Modify: `tests/test_postprocess.py`
- Modify: `src/ct_eus_vessel/postprocess.py` only if needed by the failing test.

- [ ] **Step 1: Add apex window regression**

Add:

```python
def test_apex_subsurface_cleanup_expanded_window_removes_sheet_but_preserves_protected_trunk() -> None:
    cleanup = postprocess.apply_apex_subsurface_cleanup
    multilabel = np.zeros((20, 28, 28), dtype=np.uint8)
    confidence = np.zeros(multilabel.shape, dtype=np.float32)
    liver = np.zeros(multilabel.shape, dtype=bool)
    liver[2:19, 4:24, 4:24] = True
    protected = np.zeros(multilabel.shape, dtype=bool)
    anchor = np.zeros(multilabel.shape, dtype=bool)

    sheet = [(16, y, x) for y in range(10, 14) for x in range(12, 17)]
    trunk = [(15, 18, x) for x in range(8, 23)]
    for index in sheet + trunk:
        multilabel[index] = 3
        confidence[index] = 0.55
    for index in trunk:
        protected[index] = True
    anchor[15, 18, 10] = True

    metrics = cleanup(
        multilabel,
        confidence,
        anchor_mask=anchor,
        liver_mask=liver,
        spacing_xyz=(1.0, 1.0, 1.0),
        enabled=True,
        apex_fraction=0.20,
        subsurface_min_depth_mm=2.0,
        subsurface_max_depth_mm=8.0,
        confidence_min=0.80,
        min_component_volume_mm3=8.0,
        max_component_volume_mm3=80.0,
        max_component_linearity=4.5,
        min_surface_fraction=0.30,
        anchor_dilation_mm=1.0,
        protection_mask=protected,
    )

    for index in sheet:
        assert multilabel[index] == 0
    for index in trunk:
        assert multilabel[index] == 3
    assert metrics["apex_subsurface_cleanup_voxels"] == len(sheet)
    assert metrics["apex_subsurface_cleanup_candidate_voxels"] >= len(sheet)
    assert metrics["apex_subsurface_cleanup_protected_voxels"] > 0
```

- [ ] **Step 2: Run apex cleanup test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py::test_apex_subsurface_cleanup_expanded_window_removes_sheet_but_preserves_protected_trunk
```

Expected: PASS with current function if protection mask is honored; otherwise update `apply_apex_subsurface_cleanup` minimally to honor the full protected trunk mask.

- [ ] **Step 3: Commit**

```bash
git add src/ct_eus_vessel/postprocess.py tests/test_postprocess.py
git commit -m "test: lock apex subsurface cleanup window"
```

---

### Task 6: Run Full Automated Verification

**Files:**
- No source edits unless tests fail.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=src pytest -q tests/test_config.py tests/test_postprocess.py -k "v13 or intrahepatic_trunk or apex_subsurface"
```

Expected: PASS.

- [ ] **Step 2: Run pipeline integration tests**

```bash
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py -k "v13 or v10 or v11 or v9_tube_fills_bridge"
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

```bash
PYTHONPATH=src pytest -q
```

Expected: PASS.

- [ ] **Step 4: Commit verification-only adjustments only when files changed**

```bash
git status --short
```

Expected: if `git status --short` is empty, do not create a commit. If tests required a source/test adjustment, stage only those changed files shown by `git status --short` and commit with:

```bash
git commit -m "fix: stabilize v13 trunk and apex tests"
```

---

### Task 7: Run Real CT-0021 v13

**Files:**
- Modify: `docs/ct_eus_vessel_reconstruction_iterations.md`

- [ ] **Step 1: Run real CT**

Run:

```bash
PYTHONPATH=src python -m ct_eus_vessel.cli run \
  --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' \
  --output '/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v13-trunk-connect-apex-subsurface' \
  --config config/ct0021_v13.yaml \
  --vesselness-mode slice-frangi
```

Expected:

- Output directory is created.
- `metrics_report.json` has `"label": null`.
- `guidance_source` is `"auto_totalseg_priors"`.
- Root output has `reference_ct.nrrd`, `vessel_fused_multilabel.nrrd`, `vessel_confidence.nrrd`, and `mesh/vessel_fused_slicer_ras.ply`.
- Root output has no `.nii.gz`; compatibility NIfTI files are under `compat_nifti/`.

- [ ] **Step 2: Extract acceptance metrics**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v13-trunk-connect-apex-subsurface/metrics_report.json')
summary = json.loads(path.read_text(encoding='utf-8'))
qm = summary['quality_metrics']
keys = [
    'smv_portal_bridge_repair_voxels',
    'smv_portal_bridge_repair_pairs',
    'smv_portal_bridge_repair_evidence_voxels',
    'intrahepatic_trunk_connected_before',
    'intrahepatic_trunk_connected_after',
    'intrahepatic_trunk_disconnected_components_before',
    'intrahepatic_trunk_disconnected_components_after',
    'intrahepatic_trunk_reconnect_voxels',
    'intrahepatic_trunk_reconnect_pairs',
    'apex_subsurface_cleanup_candidate_voxels',
    'apex_subsurface_cleanup_voxels',
    'apex_subsurface_cleanup_components',
    'protected_trunk_voxels',
    'outside_body_voxels',
    'voxels_removed_by_table_gate',
]
for key in keys:
    print(f'{key}: {qm.get(key)}')
print('warnings:', summary.get('warnings'))
PY
```

Acceptance gates:

- `smv_portal_bridge_repair_pairs >= 1`
- `smv_portal_bridge_repair_voxels >= 1800`
- `intrahepatic_trunk_connected_after is True`
- `intrahepatic_trunk_disconnected_components_after == 0` for components above `300 mm3`
- `intrahepatic_trunk_reconnect_voxels > 0` if `connected_before` was false
- `apex_subsurface_cleanup_candidate_voxels > 0`
- `apex_subsurface_cleanup_voxels > 0`
- `outside_body_voxels == 0`
- `voxels_removed_by_table_gate == 0`
- `warnings == []`

- [ ] **Step 3: Update iteration docs with the actual script output**

Append to `docs/ct_eus_vessel_reconstruction_iterations.md`:

```markdown
## CT-0021 v13 真实病例结果

- 输出目录：
  - `/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v13-trunk-connect-apex-subsurface`
- 本轮目标：
  - 修复肝内主血管与主干断裂。
  - 清理肝尖浅表下约 1cm 片状/破洞网状组织。
- 核心指标来自本计划 Task 7 Step 2 的脚本输出，逐项记录以下字段的真实数值：
  - `smv_portal_bridge_repair_voxels`
  - `intrahepatic_trunk_connected_before`
  - `intrahepatic_trunk_connected_after`
  - `intrahepatic_trunk_reconnect_voxels`
  - `apex_subsurface_cleanup_candidate_voxels`
  - `apex_subsurface_cleanup_voxels`
  - `protected_trunk_voxels`
  - `outside_body_voxels`
  - `warnings`
- Slicer 复核：
  - 只加载同一输出目录根部 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`。
  - 复测前删除旧 volume、segmentation、closed surface 和 model 节点。
  - 重点看肝内主血管是否和主干连续，以及肝尖浅表下约 1cm 片状/破洞网状组织是否被清理。
```

- [ ] **Step 4: Commit real-run docs**

```bash
git add docs/ct_eus_vessel_reconstruction_iterations.md
git commit -m "docs: record v13 real ct validation"
```

---

## Self-Review Checklist

- [ ] 计划覆盖问题 A：有组件级 audit，有 reconnect，有 after 指标。
- [ ] 计划覆盖问题 B：有扩大 apex window，有 protected trunk，有真实 CT 删除体素验收。
- [ ] 计划保护 v12b 已恢复的 SMV/portal bridge：验收要求 `smv_portal_bridge_repair_voxels >= 1800` 且 `pairs >= 1`。
- [ ] 计划遵守无标签边界：人工标注不进入 run，只能后续 compare。
- [ ] 计划遵守 Slicer 输出规则：复核使用根目录 NRRD 或 PLY，不使用 `compat_nifti/*.nii.gz` 生成 Slicer 3D model。
