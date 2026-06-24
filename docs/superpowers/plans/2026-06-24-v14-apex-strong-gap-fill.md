# CT-0021 v14 Apex Strong Cleanup And Gap Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build V14 so CT-0021 removes the liver-apex subsurface sheet more aggressively while filling evidence-backed short intrahepatic trunk gaps without regressing SMV reconstruction.

**Architecture:** Keep V13 component-level trunk reconnect, then add a second local evidence-fill pass for gaps inside already-connected trunk components. Scope apex protection by configuration: V13 can still use broad `protected_trunk`, while V14 uses only SMV bridge voxels plus newly repaired trunk/gap voxels so apex sheets are not overprotected.

**Tech Stack:** Python, NumPy, SciPy ndimage, SimpleITK, pytest, existing `ct_eus_vessel.postprocess` and `ct_eus_vessel.pipeline` modules.

---

### Task 1: V14 Config Baseline

**Files:**
- Create: `config/ct0021_v14.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Add this test to `tests/test_config.py`:

```python
def test_ct0021_v14_config_strengthens_apex_and_enables_gap_fill() -> None:
    config = load_config(Path("config/ct0021_v14.yaml"))
    vessel = config["vessel_extraction"]
    apex = vessel["apex_subsurface_cleanup"]
    assert apex["enabled"] is True
    assert apex["apex_fraction"] >= 0.50
    assert apex["subsurface_min_depth_mm"] <= 2
    assert apex["subsurface_max_depth_mm"] >= 18
    assert apex["min_component_volume_mm3"] <= 32
    assert apex["max_component_linearity"] >= 6.0
    assert apex["protection_source"] == "bridge_and_reconnect"

    gap = vessel["intrahepatic_trunk_gap_fill"]
    assert gap["enabled"] is True
    assert gap["max_gap_mm"] >= 12
    assert gap["max_component_volume_mm3"] >= 512
    assert gap["bridge_confidence"] >= 0.86

    bridge = vessel["smv_portal_bridge_repair"]
    assert bridge["enabled"] is True
    assert bridge["morphological_tube_fill_enabled"] is True
    assert bridge["max_gap_mm"] == 30
```

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py::test_ct0021_v14_config_strengthens_apex_and_enables_gap_fill
```

Expected: FAIL because `config/ct0021_v14.yaml` does not exist.

- [ ] **Step 3: Create V14 config**

Create `config/ct0021_v14.yaml` from V13, changing:

```yaml
vessel_extraction:
  apex_subsurface_cleanup:
    enabled: true
    apex_fraction: 0.70
    subsurface_min_depth_mm: 2
    subsurface_max_depth_mm: 18
    confidence_min: 0.86
    min_component_volume_mm3: 16
    max_component_volume_mm3: 6000
    max_component_linearity: 6.0
    min_surface_fraction: 0.20
    anchor_dilation_mm: 0
    protection_source: bridge_and_reconnect
  intrahepatic_trunk_gap_fill:
    enabled: true
    target_labels:
      - portal
      - venous
    max_gap_mm: 14
    contact_radius_mm: 1.5
    min_component_volume_mm3: 1
    max_component_volume_mm3: 800
    max_component_linearity: 8.0
    min_contact_components: 2
    bridge_confidence: 0.86
```

Keep the V13 `smv_portal_bridge_repair` block unchanged.

- [ ] **Step 4: Run green test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py::test_ct0021_v14_config_strengthens_apex_and_enables_gap_fill
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/ct0021_v14.yaml tests/test_config.py
git commit -m "test: add v14 config baseline"
```

### Task 2: Evidence-Backed Internal Trunk Gap Fill

**Files:**
- Modify: `src/ct_eus_vessel/postprocess.py`
- Modify: `tests/test_postprocess.py`

- [ ] **Step 1: Write failing postprocess test**

Add this test to `tests/test_postprocess.py`:

```python
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
    detour = [(2, y, 9) for y in range(11, 15)] + [(2, 14, x) for x in range(10, 16)] + [(2, y, 15) for y in range(11, 14)]
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
```

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py::test_intrahepatic_trunk_gap_fill_repairs_internal_candidate_gap_in_connected_component
```

Expected: FAIL with missing `apply_intrahepatic_trunk_gap_fill`.

- [ ] **Step 3: Implement minimal gap fill**

Add `_zero_trunk_gap_fill_metrics()` and `apply_intrahepatic_trunk_gap_fill()` to `src/ct_eus_vessel/postprocess.py`. The function should:

- use `_label_ids_from_names()`;
- restrict to `body_mask & ~hard_exclusion_mask & liver_mask`;
- build seed-connected target components from `trunk_seed_mask`;
- iterate candidate components from `candidate_mask & allowed & ~target`;
- reject components outside volume, max extent, or linearity limits;
- dilate each candidate by `contact_radius_mm`;
- fill only when contact with the seed-connected target has at least `min_contact_components` separated contact clusters;
- assign the majority touching label and raise confidence to `bridge_confidence`;
- return metrics:
  - `intrahepatic_trunk_gap_fill_voxels`
  - `intrahepatic_trunk_gap_fill_components`
  - `intrahepatic_trunk_gap_fill_max_gap_mm`
  - `intrahepatic_trunk_gap_fill_rejected_components`

- [ ] **Step 4: Run green test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_postprocess.py::test_intrahepatic_trunk_gap_fill_repairs_internal_candidate_gap_in_connected_component
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ct_eus_vessel/postprocess.py tests/test_postprocess.py
git commit -m "feat: fill evidence-backed intrahepatic trunk gaps"
```

### Task 3: Wire Gap Fill And Narrow Apex Protection

**Files:**
- Modify: `src/ct_eus_vessel/pipeline.py`
- Modify: `tests/test_pipeline_auto.py`

- [ ] **Step 1: Write failing pipeline test**

Add a V14 pipeline test next to the V13 test. It should monkeypatch `apply_intrahepatic_trunk_gap_fill` and `apply_apex_subsurface_cleanup`, then assert:

- SMV bridge repair runs first;
- gap fill runs before apex cleanup;
- apex cleanup receives a protection mask containing gap-filled voxels;
- apex cleanup protection mask does not protect the synthetic apex sheet;
- final fused output keeps SMV bridge and gap-filled trunk while removing apex sheet.

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py::test_run_pipeline_v14_gap_fill_protects_repair_but_not_apex_sheet
```

Expected: FAIL because pipeline does not import/call `apply_intrahepatic_trunk_gap_fill`.

- [ ] **Step 3: Implement pipeline wiring**

In `src/ct_eus_vessel/pipeline.py`:

- import `apply_intrahepatic_trunk_gap_fill`;
- initialize zero gap-fill metrics;
- before trunk reconnect, save `pre_trunk_repair_mask = fused.multilabel > 0`;
- run V13 `apply_intrahepatic_trunk_reconnect`;
- run V14 `apply_intrahepatic_trunk_gap_fill` if `vessel_cfg["intrahepatic_trunk_gap_fill"]["enabled"]`;
- compute `trunk_repair_mask = (fused.multilabel > 0) & ~pre_trunk_repair_mask`;
- support `apex_subsurface_cleanup.protection_source == "bridge_and_reconnect"` by passing `bridge_seed_mask | trunk_repair_mask`;
- add gap-fill metrics to `quality_metrics`.

- [ ] **Step 4: Run green test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py::test_run_pipeline_v14_gap_fill_protects_repair_but_not_apex_sheet
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ct_eus_vessel/pipeline.py tests/test_pipeline_auto.py
git commit -m "feat: wire trunk gap fill into v14 pipeline"
```

### Task 4: Targeted Verification And Real CT

**Files:**
- Modify: `docs/ct_eus_vessel_reconstruction_iterations.md`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py
PYTHONPATH=src pytest -q tests/test_postprocess.py -k "intrahepatic_trunk_gap_fill or intrahepatic_trunk_reconnect or apex_subsurface_cleanup"
PYTHONPATH=src pytest -q tests/test_pipeline_auto.py -k "v14 or v13 or v11 or v10 or v9_tube_fills_bridge"
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Run real CT-0021 V14**

Run:

```bash
PYTHONPATH=src python -m ct_eus_vessel.cli run \
  --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' \
  --output '/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v14-apex-strong-gap-fill' \
  --config config/ct0021_v14.yaml \
  --vesselness-mode slice-frangi
```

Expected:

- root has `reference_ct.nrrd`, `vessel_fused_multilabel.nrrd`, `vessel_confidence.nrrd`, `mesh/vessel_fused_slicer_ras.ply`;
- root has no `.nii.gz`;
- `compat_nifti/` contains compatibility NIfTI files;
- `metrics_report.json` has `label: null`, `guidance_source: auto_totalseg_priors`, `warnings: []`.

- [ ] **Step 4: Record metrics**

Append V14 results to `docs/ct_eus_vessel_reconstruction_iterations.md`, including:

- SMV bridge metrics;
- trunk reconnect and gap fill metrics;
- apex subsurface cleanup metrics;
- Slicer loading instructions.

- [ ] **Step 5: Commit and push**

```bash
git add docs/ct_eus_vessel_reconstruction_iterations.md
git commit -m "docs: record ct0021 v14 real run"
git push -u origin v14-apex-gap-fill
```

Expected: branch pushed to remote.

---

## Self-Review

- Spec coverage: covers stronger apex cleanup, narrower protection, internal trunk gap fill, SMV non-regression, real CT output documentation.
- Placeholder scan: no placeholders or TBD items.
- Type consistency: uses `apply_intrahepatic_trunk_gap_fill`, `intrahepatic_trunk_gap_fill_*` metrics, and `protection_source: bridge_and_reconnect` consistently across config, postprocess, pipeline, and docs.
