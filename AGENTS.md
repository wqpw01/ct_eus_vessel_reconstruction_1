# Agent 规则

## 适用范围

- 本文件适用于当前 CT-EUS 血管重建仓库。

## Slicer 输出防错

- 若 3D Slicer 三视图中 CT 与 labelmap 对齐，但 closed surface 或 model 偏移、跑出紫色体积框，先按 Slicer 3D 输出路径排查，不要先改分割算法。
- Slicer 3D 建模只建议使用同一输出目录根部的 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`。
- `compat_nifti/*.nii.gz` 仅用于非 Slicer 兼容分析；不要建议用它们在 Slicer 中生成 3D model 或 closed surface。
- 复测 Slicer 对齐问题前，先提醒删除旧 volume、segmentation、closed surface 和 model 节点，避免旧缓存误导判断。
- 重新生成输出后，根目录不应残留 `.nii.gz`；兼容 NIfTI 应放在 `compat_nifti/`。

## 多期覆盖防错

- 若输出在 S 方向上方整体缺失，先检查 reference 是否被固定为 portal-only 或其他单一期相。
- 多期 CT 覆盖范围不一致时，输出网格应使用 union reference，并生成 composite reference CT；不要用单一期相 reference 裁剪融合 mask。
