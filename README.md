# CT-EUS Vessel Reconstruction

Research pipeline for extracting contrast-enhanced thoracoabdominal vessels from multi-phase CT for CT-EUS localization.

## What v1 Does

- Indexes DICOM series by `SeriesInstanceUID` and metadata, not filenames.
- Selects thin soft-kernel contrast series and automatically maps arterial, portal venous, and venous-support phases from DICOM metadata and acquisition order.
- Uses image-derived dynamic HU windows, slice-wise Frangi vesselness, TotalSegmentator organ/bone priors, high-HU bone-like exclusion, image-derived anchors, and 3D connected-component filtering.
- Writes Slicer-friendly NRRD CT/masks, compatibility NIfTI masks, confidence volume, bbox JSON, QC PNGs, and optional PLY mesh.

## Quick Run

```bash
cd /home/zyt/ct_eus_vessel_reconstruction
PYTHONPATH=src python -m ct_eus_vessel.cli index \
  --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' \
  --json series_index.json

PYTHONPATH=src python -m ct_eus_vessel.cli run \
  --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' \
  --output output/CT-0021-auto-totalseg-v1
```

Use `--skip-frangi` for fast I/O smoke tests only. It relies mostly on HU and has many false positives.

The default run no longer requires `pseudo_label-.nii`. It runs TotalSegmentator once on the composed reference CT, caches the multilabel prior at `totalseg/roi_subset_multilabel.nii.gz`, and reuses it on later runs. Use `--force-totalseg` to refresh that cache, `--totalseg-device cpu` to force CPU, or `--label <path>` only for legacy/manual-label comparison.

For 3D Slicer, prefer the NRRD pair from the same output directory:

- `reference_ct.nrrd`
- `vessel_fused_multilabel.nrrd`
- optional model: `mesh/vessel_fused_slicer_ras.ply`

The NRRD CT and NRRD labelmap are written as a Slicer-native RAS-space pair with identical size, spacing, origin, and direction. The PLY model is exported in Slicer RAS physical coordinates. Do not use older `vessel_fused.ply` files from previous experimental output folders; those were local voxel-space meshes and will appear offset from the CT. The `.nii.gz` files are kept under `compat_nifti/` for non-Slicer compatibility, but NRRD is the recommended Slicer path because some model-generation paths ignore NIfTI qform/sform transforms.

## 常见错误 / 排错

### Slicer 三视图对齐，但 3D 模型偏移

现象：在 3D Slicer 的 axial/sagittal/coronal 三视图中，CT 和 labelmap 看起来完全对齐；但生成 closed surface 或 model 后，3D 模型偏离 CT，甚至跑出紫色体积框。

默认先排查 Slicer 3D 输出路径，不要先怀疑分割本身。已确认的常见原因包括：用 `compat_nifti/` 下的 `.nii.gz` 生成 3D、Slicer 的某些 3D 模型生成路径忽略 NIfTI qform/sform、加载了旧的 closed surface/model 缓存，或误用了历史输出里的 voxel-space PLY。

正确做法：

- 生成 3D 时只使用同一输出目录根部的 `reference_ct.nrrd` 和 `vessel_fused_multilabel.nrrd`。
- 如需直接加载模型，使用 `mesh/vessel_fused_slicer_ras.ply`。
- `compat_nifti/` 只用于非 Slicer 兼容分析，不用于 Slicer 3D 建模。
- 复测前删除 Slicer 场景里的旧 volume、segmentation、closed surface 和 model 节点，避免复用缓存。

### S 方向上方整体缺失

现象：模型不是局部断裂，而是在 S 方向上方整体没有内容。

默认先检查输出 reference 是否被固定成单一期相，尤其是 portal-only reference。多期 CT 覆盖范围不一致时，输出网格必须使用 union reference，并用 composite reference CT 补齐 portal 覆盖不到但 venous/support 覆盖到的区域，避免把真实 mask 裁掉。

## Outputs

- `compat_nifti/artery_candidate.nii.gz`
- `artery_candidate.nrrd`
- `compat_nifti/portal_vein_candidate.nii.gz`
- `portal_vein_candidate.nrrd`
- `compat_nifti/systemic_vein_candidate.nii.gz`
- `systemic_vein_candidate.nrrd`
- `compat_nifti/vessel_fused_multilabel.nii.gz`
- `vessel_fused_multilabel.nrrd`
- `compat_nifti/vessel_confidence.nii.gz`
- `vessel_confidence.nrrd`
- `reference_ct.nrrd`
- `bbox.json`
- `metrics_report.json`
- `qc/overlay_axial.png`
- `qc/mask_mip_axial.png`
- `mesh/vessel_fused_slicer_ras.ply` unless `--skip-mesh`

## Notes

- The default path is label-free. A manual weak label can still be supplied with `--label` for legacy comparison, but it is no longer searched for or required.
- Default `slice-frangi` is the local-friendly vesselness backend. Full `frangi3d` is available through `--vesselness-mode frangi3d` but used 10GB+ RAM on CT-0021 and should be paired with ROI crop or a larger server.
- TotalSegmentator is integrated as a cached organ/bone-prior step. The default config uses liver, spleen, kidneys, stomach, pancreas, and duodenum as soft penalties, and L1-L5 vertebrae as hard exclusions.

## Validation Baseline

On CT-0021, the previous label-guided baseline selected:

- arterial: `...172466`
- portal: `...182466`
- venous-support: `...14300`

That baseline wrote results to:

```text
/home/zyt/ct_eus_vessel_reconstruction/output/CT-0021-clean-v3-nrrd
```

The recommended label-free output path for a fresh run is:

```text
/home/zyt/ct_eus_vessel_reconstruction/output/CT-0021-auto-totalseg-v1
```

A label-free smoke run with `--skip-frangi --skip-mesh --max-series 3` wrote to:

```text
/home/zyt/ct_eus_vessel_reconstruction/output/CT-0021-auto-smoke
```

It selected the same arterial, portal, and venous-support series listed above, with `guidance_source` set to `auto_totalseg_priors`.
