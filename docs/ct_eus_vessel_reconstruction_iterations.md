# CT-EUS 血管重建迭代回顾

日期：2026-06-11

## 使用边界

- 无标签重建运行时不得使用人工标注结果。
- 人工标注目录只允许在重建完成后用于 `compare` 评估、误差统计和回顾分析。
- Slicer 复测优先加载输出根目录的 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`。
- `compat_nifti/*.nii.gz` 仅用于兼容分析，不用于 Slicer 生成 3D model 或 closed surface。

## 本轮主要改进

### 1. 去除体外平台和骨性高密度误检

- 在无标签路径中引入 body mask 与 hard exclusion 约束。
- 修复后 `CT-0021-fixed` 对比旧结果：
  - fused 体素：`769,546 -> 56,931`
  - body 外体素：`0`
  - hard skeleton overlap：`0`
  - table-like：`0`

### 2. 多期覆盖与 Slicer 输出防错

- 多期 CT 覆盖范围不一致时，输出使用 union reference。
- 根目录只保留 Slicer 友好的 NRRD 和 PLY，兼容 NIfTI 放入 `compat_nifti/`。
- 复测 Slicer 错位时，先清空旧 volume、segmentation、closed surface、model 节点，避免旧缓存误导。

### 3. 新增 compare 评估能力

- 新增 `ct-eus-vessel compare`，支持 reference/candidate 最近邻重采样后计算：
  - overall precision / recall / Dice
  - by-label arterial / portal / venous
  - liver-stratified 指标
- `CT-0021-fixed` 与人工参考相比：
  - overall recall：`4.41%`
  - overall Dice：`0.080`
  - 肝内 recall：`0.09%`

### 4. 肝内血管恢复 v1

- 新增 intrahepatic recovery：
  - liver ROI
  - 肝区加权 local background
  - local contrast 与 relaxed vesselness
  - 组件过滤
  - final anchor 后保留已通过过滤的肝内恢复 mask
- `CT-0021-intrahepatic-v1` 结果：
  - fused：`371,738`
  - 肝内恢复贡献：`313,793`
  - body 外体素：`0`
  - 肝内 overall recall：`76.11%`
  - 肝内 precision：`30.21%`
  - 肝内 Dice：`0.432`
- 主要问题：
  - 肝内 venous 恢复较好，但带入较多肝表面和血管周围组织。
  - portal 几何大量被 venous phase 捕获，但按标签仍多被标成 venous。
  - TotalSegmentator vessel anchors 覆盖肝外主干，但 v1 只用于筛选，没有并入最终输出。

### 5. 主干补全与肝表面 pruning v2

- 新增 `src/ct_eus_vessel/postprocess.py`：
  - `inject_totalseg_vessel_anchors`
  - `apply_liver_surface_recovery_gate`
- 默认配置：
  - `include_totalseg_vessel_anchors_in_output: true`
  - `totalseg_vessel_anchors_include_liver: false`
  - `surface_prune_enabled: true`
  - `surface_prune_depth_mm: 5`
  - `surface_prune_confidence_min: 0.75`
- 设计意图：
  - TotalSegmentator 主干 anchor 只补肝外和肝门主干，不作为肝内细分支来源。
  - surface gate 只作用于 `intrahepatic_recovery_mask`，不影响原始 phase candidate 与肝外主干 anchor。

## CT-0021 v2 结果

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v2-trunk-surface-prune`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v2-trunk-surface-prune`

重建质量指标：

- fused：`706,431`
- intrahepatic surface pruned：`141,718`
- TotalSeg anchor output：`497,741`
- body 外体素：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无

与人工参考评估：

- overall precision：`66.25%`
- overall recall：`82.98%`
- overall Dice：`0.737`
- arterial recall：`84.03%`
- portal recall：`50.79%`
- venous recall：`83.48%`
- 肝内 precision：`52.01%`
- 肝内 recall：`71.85%`
- 肝内 Dice：`0.603`

相对 v1 的主要变化：

- overall Dice：`0.256 -> 0.737`
- overall recall：`21.20% -> 82.98%`
- 肝内 precision：`30.21% -> 52.01%`
- 肝内 Dice：`0.432 -> 0.603`
- 肝外主干由 TotalSegmentator anchors 补回，portal/venous/arterial 主干召回明显提升。

### v2 后剩余问题

- 部分肝内 `Segment_2` 仍会被重建成 `Segment_3`。
- 肝脏内部血管周围还有一部分其余组织没有清理干净。
- 部分肝脏表面浅层残留仍未剔除。

### v3 设计决策

- 人工标注仍然只用于重建完成后的 `compare` 评估，不参与任何无标签重建输入。
- 保持 v2 基线逻辑不变，仅通过独立覆盖配置 `config/ct0021_v3.yaml` 打开两项 v3 行为：
  - `portal_from_venous_relabel`
  - `final_liver_surface_cleanup`
- v3 阈值：
  - `portal_hu - venous_hu >= 30`
  - `depth <= 8 mm`
  - `confidence < 0.78`
- 接入顺序：
  - 先保留 v2 的 intrahepatic recovery 与首次 surface prune
  - 再执行 v3 的 final liver surface cleanup
  - 再执行 v3 的 portal-from-venous relabel
  - 最后再执行 TotalSegmentator 肝外主干 anchor 注入和 table gate

### v3 实现内容

- `src/ct_eus_vessel/postprocess.py`
  - 新增 `apply_final_liver_surface_cleanup`
  - 新增 `apply_portal_from_venous_relabel`
- `src/ct_eus_vessel/pipeline.py`
  - 新增 portal/venous 相强度与 coverage 采样
  - 将 v3 两步后处理接入最终输出路径
  - 新增 `quality_metrics` 字段：
    - `portal_relabel_voxels`
    - `portal_relabel_by_reason`
    - `final_liver_surface_cleanup_voxels`
    - `final_liver_surface_cleanup_by_label`
- `config/default.yaml`
  - 保留 v2 基线，默认关闭 v3 行为
- `config/ct0021_v3.yaml`
  - 仅为 CT-0021 开启 v3 阈值覆盖

## CT-0021 v3 结果

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v3-portal-relabel-surface-cleanup`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v3-portal-relabel-surface-cleanup`

v3 后处理统计：

- fused：`662,694`
- intrahepatic surface pruned（v2 既有 gate）：`141,718`
- final liver surface cleanup（v3 新增）：`43,737`
- portal relabel voxels（v3 新增）：`7,593`
- portal relabel eligible venous liver voxels：`127,757`
- portal missing coverage：`50,162`
- insufficient HU margin：`70,002`
- TotalSeg anchor output：`497,741`
- body 外体素：`0`
- hard skeleton overlap：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无

与人工参考评估：

- overall precision：`69.25%`
- overall recall：`81.37%`
- overall Dice：`0.748`
- arterial precision / recall / Dice：`69.68% / 84.03% / 0.762`
- portal precision / recall / Dice：`61.31% / 55.30% / 0.581`
- venous precision / recall / Dice：`66.53% / 81.37% / 0.732`
- 肝内 precision：`62.67%`
- 肝内 recall：`64.56%`
- 肝内 Dice：`0.636`

相对 v2 的主要变化：

- overall precision：`66.25% -> 69.25%`
- overall recall：`82.98% -> 81.37%`
- overall Dice：`0.737 -> 0.748`
- portal recall：`50.79% -> 55.30%`
- 肝内 precision：`52.01% -> 62.67%`
- 肝内 recall：`71.85% -> 64.56%`
- 肝内 Dice：`0.603 -> 0.636`

几何与 Slicer 复核：

- 当前 v3 输出目录内部的 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`：
  - size / spacing / origin / direction 全部一致
- `mesh/vessel_fused_slicer_ras.ply` 已生成
- `compare` 报告中的 reference/candidate 目录几何差异来自人工参考目录与当前无标签输出目录本身网格不同，不是 v3 目录内部 CT/labelmap 配对错误
- 复测 Slicer 仍建议只加载根目录 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`

## 后续关注点

- 肝内 portal 标签仍偏弱，主要问题是 portal 树常被晚期相捕获并标为 venous。
- 如果需要进一步提升 portal 标签，应采用多期差分或解剖连通约束，不能把人工标注作为无标签重建输入。
- 若 v2 主干 anchor 显得过粗，下一轮优先收紧 anchor 注入范围到肝外加肝门小范围，而不是关闭主干补全。

## CT-0021 v4 规划

### 目标

- 保留 v3 的整体提升，不覆盖 v3 输出目录。
- 修复当前两个主要问题：
  - 肝门/主干连接处被清断。
  - 肝内深部仍有一部分 `Segment_3` 组织和肝尖浅表残留。

### 设计原则

- 人工标注继续只用于重建后的 `compare`，不进入无标签重建。
- 保持 `CT-0021-v3-portal-relabel-surface-cleanup` 不变，v4 使用新的配置和新的输出目录。
- 以“保连接优先”为主，清理只打远离肝门主干的残留。

### 拟定修改

- 将 `portal_from_venous_relabel` 前移到最终清理之前，避免清理先把桥段删掉。
- 增加肝门保护区：
  - 以 portal / venous anchors 的组合距离为准
  - 默认保护半径：`12 mm`
  - 保护区内不做最终清理
- 放宽肝门保护区内的 portal 重标条件：
  - 保护区内使用 `portal_hu - venous_hu >= 20`
  - 保护区外保持 `>= 30`
- 将最终清理拆成两步：
  - 表面清理：只删 `depth <= 8 mm`、`confidence < 0.78`、且不在肝门保护区内的 `Segment_3`
  - 深部清理：只删 `anchor_distance > 12 mm`、`confidence < 0.78`、且仍为 `Segment_3` 的远端残留
- 保留 TotalSegmentator 肝外主干 anchor 注入和 table gate 不变。

### 新增配置与输出

- 新配置文件：`config/ct0021_v4.yaml`
- 新输出目录：
  - `CT-0021-v4-bridge-protect-deep-cleanup`
- 新增质量指标：
  - `hilar_protection_voxels`
  - `portal_relabel_bridge_voxels`
  - `final_surface_cleanup_voxels`
  - `deep_liver_cleanup_voxels`
  - `deep_liver_cleanup_by_label`

### 测试与验收

- 新增纯函数测试：
  - 保护区内的桥段不会被最终清理删掉。
  - 保护区内的 venous 体素在满足更宽松阈值时可重标为 portal。
  - 保护区外的深部 `Segment_3` 残留会被清掉。
- 新增 pipeline 集成测试：
  - 肝门连接不断。
  - 肝尖表面残留进一步下降。
  - `outside_body_voxels == 0`
  - `hard overlap == 0`
  - `table gate == 0`
- 真实病例只在 v4 输出目录完成后再做 compare，不覆盖 v3 目录。

### v4 实现内容

- `src/ct_eus_vessel/postprocess.py`
  - 新增 `build_hilar_protection_mask`
  - `apply_portal_from_venous_relabel` 支持保护区内更宽松阈值，并统计 `portal_relabel_bridge_voxels`
  - `apply_final_liver_surface_cleanup` 支持 `protection_mask`，保护区内不清理
  - 新增 `apply_deep_liver_cleanup`
- `src/ct_eus_vessel/pipeline.py`
  - 检测 `hilar_protection + deep_liver_cleanup` 时进入 v4 顺序：
    - 先进行 portal-from-venous relabel
    - 再进行受保护的 surface cleanup
    - 最后进行 deep liver cleanup
  - 新增质量指标：
    - `hilar_protection_voxels`
    - `portal_relabel_bridge_voxels`
    - `final_surface_cleanup_voxels`
    - `deep_liver_cleanup_voxels`
    - `deep_liver_cleanup_by_label`
- `config/ct0021_v4.yaml`
  - 启用 v4 覆盖阈值：
    - 肝门保护半径：`12 mm`
    - 保护区内 portal 重标阈值：`20 HU`
    - 保护区外 portal 重标阈值：`30 HU`
    - 表面清理：`depth <= 8 mm` 且 `confidence < 0.78`
    - 深部清理：`anchor_distance > 12 mm` 且 `confidence < 0.78`

## CT-0021 v4 结果

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v4-bridge-protect-deep-cleanup`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v4-bridge-protect-deep-cleanup`

v4 后处理统计：

- fused：`602,180`
- intrahepatic surface pruned（v2 既有 gate）：`141,718`
- hilar protection voxels：`486,315`
- portal relabel voxels：`11,745`
- portal relabel bridge voxels：`3,331`
- final liver surface cleanup：`30,456`
- deep liver cleanup：`73,783`
- TotalSeg anchor output：`497,729`
- body 外体素：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无

与人工参考评估：

- overall precision：`69.73%`
- overall recall：`74.46%`
- overall Dice：`0.720`
- arterial precision / recall / Dice：`69.68% / 84.03% / 0.762`
- portal precision / recall / Dice：`59.31% / 56.86% / 0.581`
- venous precision / recall / Dice：`69.61% / 68.94% / 0.693`
- 肝内 precision：`61.11%`
- 肝内 recall：`33.25%`
- 肝内 Dice：`0.431`

相对 v3 的主要变化：

- overall Dice：`0.748 -> 0.720`
- overall recall：`81.37% -> 74.46%`
- portal recall：`55.30% -> 56.86%`
- venous recall：`81.37% -> 68.94%`
- 肝内 recall：`64.56% -> 33.25%`
- 肝内 Dice：`0.636 -> 0.431`

### v4 结论

- v4 代码路径与输出格式验证通过，但当前 `deep_liver_cleanup` 阈值过强，真实病例召回明显下降。
- v4 输出可以作为失败/过清理对照保留，不建议替代 v3 作为当前最佳无标签结果。
- 当前推荐基线仍是 `CT-0021-v3-portal-relabel-surface-cleanup`。
- 下一轮若继续改，应先削弱或拆分 deep cleanup：
  - 提高 `confidence_min` 清理门槛的保守性，或只清孤立小组件。
  - 将 anchor distance 与连通性、局部管状度结合，不要单独按远离 anchor 删除。
  - 继续坚持人工标注只用于 compare，不进入无标签重建输入。

## CT-0021 v5 规划与实现

### 目标

- 保留 v4 的肝门/主干连接保护与保护区内 portal 重标。
- 撤回 v4 的全局 `deep_liver_cleanup`，避免再次把真实肝内 venous 分支按“远离 anchor + 低 confidence”一刀切删除。
- 深部清理只针对孤立、低置信、非管状的小块残留；细长连通血管分支默认保留。
- 使用新输出目录，不覆盖 v3/v4。

### 证据依据

- v4 相比 v3 少 `73,795` 个体素。
- 其中 `73,783` 个在肝内，且几乎全部来自 v3 的 venous label。
- 被删除肝内体素中 `45,624` 个与人工参考血管重合。
- 肝内 recall 从 v3 的 `64.56%` 下降到 v4 的 `33.25%`。
- 因此问题不是肝门保护，而是 `deep_liver_cleanup` 规则过粗。

### v5 实现内容

- `config/ct0021_v5.yaml`
  - 保留 `hilar_protection`、`portal_from_venous_relabel`、受保护的 `final_liver_surface_cleanup`。
  - 关闭 `deep_liver_cleanup`。
  - 启用 `isolated_liver_blob_cleanup`：
    - `max_component_volume_mm3: 32`
    - `max_component_elongation: 2.0`
    - `confidence_min: 0.78`
    - `anchor_dilation_mm: 3`
- `src/ct_eus_vessel/postprocess.py`
  - 新增 `apply_isolated_liver_blob_cleanup`
  - 只删除 liver 内低置信 venous 小组件。
  - 保留 elongation 高的细长分支。
  - 保留与 portal/venous anchor 邻近或位于 hilar protection 内的组件。
- `src/ct_eus_vessel/pipeline.py`
  - 当启用 `hilar_protection + isolated_liver_blob_cleanup` 时沿用 v4 顺序：
    - 先 portal-from-venous relabel
    - 再受保护 surface cleanup
    - 最后 isolated blob cleanup
  - 新增质量指标：
    - `isolated_blob_cleanup_voxels`
    - `isolated_blob_cleanup_components`
    - `isolated_blob_cleanup_by_label`

### 测试

- 新增纯函数测试：
  - 孤立低置信 blob 会被删除。
  - 细长低置信 venous 分支不会被删除。
  - protection mask 内组件不会被删除。
- 新增 pipeline 集成测试：
  - 肝门桥段保留并可重标为 portal。
  - 浅表低置信 venous 仍由 surface cleanup 删除。
  - 深部细长 venous 分支保留。
  - 深部孤立 blob 删除。
  - `deep_liver_cleanup_voxels == 0`。
- 全量测试：`67 passed`。

### 待真实病例验证

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v5-conservative-bridge-blob-cleanup`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v5-conservative-bridge-blob-cleanup`

v5 后处理统计：

- fused：`665,740`
- intrahepatic surface pruned：`141,720`
- hilar protection voxels：`486,315`
- portal relabel voxels：`11,745`
- portal relabel bridge voxels：`3,331`
- final liver surface cleanup：`30,455`
- deep liver cleanup：`0`
- isolated blob cleanup：`10,236`
- isolated blob cleanup components：`2,826`
- TotalSeg anchor output：`497,741`
- body 外体素：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无

与人工参考评估：

- overall precision：`69.33%`
- overall recall：`81.84%`
- overall Dice：`0.751`
- arterial precision / recall / Dice：`69.68% / 84.03% / 0.762`
- portal precision / recall / Dice：`59.31% / 56.86% / 0.581`
- venous precision / recall / Dice：`66.93% / 81.60% / 0.735`
- 肝内 precision：`63.24%`
- 肝内 recall：`66.69%`
- 肝内 Dice：`0.649`

相对 v4 的主要变化：

- overall Dice：`0.720 -> 0.751`
- overall recall：`74.46% -> 81.84%`
- venous recall：`68.94% -> 81.60%`
- 肝内 recall：`33.25% -> 66.69%`
- 肝内 Dice：`0.431 -> 0.649`

验收重点：

- 肝内 recall 应明显高于 v4，并尽量回到 v3 附近。
- portal recall 保留 v4 的桥段重标收益。
- `outside_body_voxels == 0`。
- `voxels_removed_by_table_gate == 0`。
- 根目录无 `.nii.gz`。
- Slicer 复测仍只使用根目录 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`，或 `mesh/vessel_fused_slicer_ras.ply`。

## CT-0021 v6 规划与实现

### 目标

- 在 v5 基础上继续清理用户在 Slicer 中指出的肝尖、肝表面和部分肝内非血管残留。
- 取舍略偏向清理干净，但仍保留 v5 已恢复的主干、肝门桥段和细长肝内分支。
- 不恢复 v4 的全局 `deep_liver_cleanup`，避免按“远离 anchor + 低 confidence”误删真实肝内血管。

### v6 实现内容

- `config/ct0021_v6.yaml`
  - 保留 `hilar_protection`、保护区内 `portal_from_venous_relabel` 和受保护的 `final_liver_surface_cleanup`。
  - 继续关闭 `deep_liver_cleanup`。
  - 略加强 `isolated_liver_blob_cleanup`：
    - `max_component_volume_mm3: 48`
    - `confidence_min: 0.80`
    - `anchor_dilation_mm: 4`
  - 新增 `apex_surface_morph_cleanup`：
    - `surface_depth_mm: 10`
    - `apex_fraction: 0.12`
    - `confidence_min: 0.82`
    - `max_component_volume_mm3: 96`
    - `max_component_elongation: 2.4`
    - `anchor_dilation_mm: 4`
- `src/ct_eus_vessel/postprocess.py`
  - 新增 `apply_apex_surface_morph_cleanup`
  - 只作用于 liver 内低置信 venous。
  - 候选位置限制在肝表面浅层或肝顶区域。
  - 通过组件体积和 PCA 线性度清理偏片状/团块状残留，同时保留线性度高的细长分支。
  - 保留 hilar protection 与 portal/venous anchor dilation 内组件。
- `src/ct_eus_vessel/pipeline.py`
  - 在 v5 的 isolated blob cleanup 之后接入 apex/surface morph cleanup。
  - 新增质量指标：
    - `apex_surface_cleanup_voxels`
    - `apex_surface_cleanup_components`
    - `apex_surface_cleanup_by_label`
    - `apex_surface_cleanup_by_region`

### 测试

- 新增纯函数测试：
  - 肝顶低置信团块会删除。
  - 肝表面低置信片状残留会删除。
  - 细长低置信 venous 分支保留。
  - hilar protection 和 anchor dilation 内组件保留。
  - 高置信 surface venous 与 portal label 不受影响。
- 新增 pipeline 集成测试：
  - 肝门桥段保留并重标为 portal。
  - final surface cleanup 仍先清低置信浅表点。
  - isolated blob cleanup 继续清深部孤立小块。
  - apex/surface morph cleanup 清肝顶和表面残留。
  - 深部细长 venous 分支保留。
  - `deep_liver_cleanup_voxels == 0`。
- 局部测试：
  - `tests/test_postprocess.py`：`10 passed`
  - `tests/test_config.py`：`6 passed`
  - `tests/test_pipeline_auto.py`：`9 passed`

### 真实病例验证

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v6-apex-surface-morph-cleanup`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v6-apex-surface-morph-cleanup`

v6 后处理统计：

- fused：`660,273`
- intrahepatic surface pruned：`141,720`
- hilar protection voxels：`486,315`
- portal relabel voxels：`11,745`
- portal relabel bridge voxels：`3,331`
- final liver surface cleanup：`30,455`
- deep liver cleanup：`0`
- isolated blob cleanup：`9,666`
- isolated blob cleanup components：`3,380`
- apex/surface morph cleanup：`6,037`
- apex/surface morph cleanup components：`1,323`
- apex/surface morph cleanup by region：apex `753`，surface `5,284`
- TotalSeg anchor output：`497,741`
- body 外体素：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无

与人工参考评估：

- overall precision：`69.88%`
- overall recall：`81.80%`
- overall Dice：`0.754`
- portal precision / recall / Dice：`59.31% / 56.86% / 0.581`
- venous precision / recall / Dice：`67.99% / 81.55% / 0.742`
- 肝内 precision：`65.83%`
- 肝内 recall：`66.53%`
- 肝内 Dice：`0.662`

相对 v5 的主要变化：

- fused：`665,740 -> 660,273`
- overall precision：`69.33% -> 69.88%`
- overall recall：`81.84% -> 81.80%`
- overall Dice：`0.751 -> 0.754`
- 肝内 precision：`63.24% -> 65.83%`
- 肝内 recall：`66.69% -> 66.53%`
- 肝内 Dice：`0.649 -> 0.662`

### v6 结论

- v6 相比 v5 更干净，且没有出现 v4 的肝内召回崩塌。
- portal 指标与 v5 保持一致，说明保护区内桥段重标没有被新清理破坏。
- venous 与肝内 precision 提升明显，recall 仅轻微下降，符合“清得干净稍微重要一点”的取舍。
- `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd` 的 size / spacing / origin / direction 全部一致。
- `mesh/vessel_fused_slicer_ras.ply` 已生成。
- Slicer 复测仍只使用根目录 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`，或 `mesh/vessel_fused_slicer_ras.ply`。

### 性能修复

- 真实病例验证时发现两个既有组件循环在大体积数据上会反复扫描全数组：
  - `recover_intrahepatic_vessels`
  - `apply_isolated_liver_blob_cleanup`
- 已改为基于 `ndi.find_objects` 的局部 bbox 视图统计，筛选规则不变。
- 全量测试：`71 passed`。

## CT-0021 v7 规划与实现

### 背景

用户对 v6 的 Slicer/PPT 评估结论：

- v6 整体血管树可接受，后续配准前可能还需要表面平滑。
- 动脉期识别仍弱，但本轮不把动脉分类大改作为第一优先级。
- 主要新增问题是肝外右下方/器官外围仍有团块状伪血管。
- SMV/portal 主干存在断续，需要把 SMV 与 portal vein / splenic vein 一起作为 portal 系统保护对象。

用户确认 v7 优先级：

- 第一优先级：清理肝外右下方这类外周团块。
- 硬约束：SMV / portal 连续性不能被进一步破坏。

### v7 实现内容

- `config/ct0021_v7.yaml`
  - 继承 v6 的肝门保护、portal 重标、肝表面清理、isolated blob cleanup 与 apex/surface morph cleanup。
  - 继续关闭 `deep_liver_cleanup`，避免回到 v4 的过清理。
  - 新增 `smv_portal_bridge_protection`：
    - `enabled: true`
    - `distance_mm: 22`
  - 新增 `outer_peripheral_blob_cleanup`：
    - `enabled: true`
    - `max_component_volume_mm3: 160`
    - `max_component_linearity: 2.2`
    - `confidence_min: 0.82`
    - `anchor_dilation_mm: 6`

- `src/ct_eus_vessel/postprocess.py`
  - 新增 `build_smv_portal_protection_mask`：
    - 从 TotalSegmentator 的 portal anchor（`portal_vein_and_splenic_vein`）出发扩张保护区。
    - 保护区限制在 body 内且排除 hard exclusion。
  - 新增 `apply_outer_peripheral_blob_cleanup`：
    - 只作用于 liver 外、body 内、非 hard exclusion、非保护区的低置信血管标签。
    - 按组件体积和 PCA 线性度删除小团块/非管状外周残留。
    - 保留 SMV/portal 保护区、anchor dilation 内组件和线性度高的外周血管样分支。

- `src/ct_eus_vessel/pipeline.py`
  - 构建 `smv_portal_protection_mask` 并与 `hilar_protection_mask` 合并为后处理保护区。
  - portal-from-venous relabel 的 eligible 区域从 liver 内扩展到 `liver | smv_portal_protection_mask`，用于保护并重标 SMV/portal 桥段。
  - 在 TotalSeg 肝外主干 anchor 注入前执行 `outer_peripheral_blob_cleanup`。
  - 新增质量指标：
    - `smv_portal_protection_voxels`
    - `outer_peripheral_cleanup_voxels`
    - `outer_peripheral_cleanup_components`
    - `outer_peripheral_cleanup_by_label`

### 测试

- 新增纯函数测试：
  - SMV/portal 保护区只在 body 内扩张，且排除 hard exclusion。
  - 外周清理删除 liver 外低置信小团块。
  - 外周清理保留 liver 内组件、SMV/portal 保护区组件、hard exclusion、body 外组件、高置信组件和线性血管样分支。
- 新增配置测试：
  - `ct0021_v7.yaml` 保留 v6 行为并启用 `smv_portal_bridge_protection` 与 `outer_peripheral_blob_cleanup`。
- 新增 pipeline 集成测试：
  - SMV/portal bridge 保留并重标为 portal。
  - liver 外外周 portal/venous 小团块被删除。
  - liver 内分支和外周线性分支保留。
  - v4/v5/v6 的 `portal_relabel_bridge_voxels` 指标不回退。

### 真实病例验收

输出目录：

`C:\Users\zhangyutang\Desktop\CT-EUS血管重建结果\无标签\CT-0021-v7-outer-peripheral-cleanup`

WSL 路径：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v7-outer-peripheral-cleanup`

验收重点：

- Slicer 中肝外右下方/器官外围团块明显少于 v6。
- SMV / portal 主干不能比 v6 更断。
- 肝尖、肝表面清理水平不低于 v6。
- `outside_body_voxels == 0`。
- `voxels_removed_by_table_gate == 0`。
- `deep_liver_cleanup_voxels == 0`。
- 根目录无 `.nii.gz`。
- `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd` 几何一致。
- 人工标注仍只用于 compare 和复盘，不传入 `run_pipeline`。

### v7 真实病例结果

v7 后处理统计：

- fused：`659,798`
- hilar protection voxels：`486,315`
- SMV/portal protection voxels：`2,033,863`
- portal relabel voxels：`12,616`
- portal relabel bridge voxels：`6,389`
- final liver surface cleanup：`30,317`
- isolated blob cleanup：`9,771`
- apex/surface morph cleanup：`5,906`
- outer peripheral cleanup：`927`
- outer peripheral cleanup components：`32`
- outer peripheral cleanup by label：
  - arterial：`855`
  - portal：`23`
  - venous：`49`
- deep liver cleanup：`0`
- body 外体素：`0`
- table gate 移除：`0`
- 根目录 `.nii.gz`：无
- `mesh/vessel_fused_slicer_ras.ply`：已生成

与人工参考评估：

- overall precision：`69.88%`
- overall recall：`81.75%`
- overall Dice：`0.753`
- portal precision / recall / Dice：`58.92% / 57.17% / 0.580`
- venous precision / recall / Dice：`68.12% / 81.52% / 0.742`
- 肝内 precision：`65.74%`
- 肝内 recall：`66.52%`
- 肝内 Dice：`0.661`

相对 v6 的主要变化：

- fused：`660,273 -> 659,798`
- portal relabel bridge voxels：`3,331 -> 6,389`
- final liver surface cleanup：`30,455 -> 30,317`
- isolated blob cleanup：`9,666 -> 9,771`
- apex/surface morph cleanup：`6,037 -> 5,906`
- 新增 outer peripheral cleanup：`927`
- overall Dice：`0.754 -> 0.753`
- portal recall：`56.86% -> 57.17%`
- venous Dice：`0.742 -> 0.742`
- 肝内 Dice：`0.662 -> 0.661`

### v7 结论

- 用户复查指出 v7 改进不大：新增外周清理在视觉目标上接近既有肝尖/肝表面清理，真实目标外周团块仍未清掉。
- 复盘确认 v7 的 `outer_peripheral_blob_cleanup` 只删除 `810-927` 量级的小碎片，最大删除组件约 `223` 体素；但 v7 假阳性里仍存在数千到数万体素的大外周组件。
- 复盘确认 SMV/portal 仍断裂：v7 portal 最大组件约 `50,932` 体素，第二大 portal 组件约 `10,606` 体素，最近 gap 约 `22.11 mm`。
- 关键机制问题：v7 外周清理发生在 TotalSeg vessel anchor 注入前，无法审核注入后 `confidence=1.0` 的外周 anchor 团块；SMV/portal 保护只是保护和局部重标，没有显式桥接。
- Slicer 复测仍只使用根目录 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`，或 `mesh/vessel_fused_slicer_ras.ply`。

## CT-0021 v8 规划与实现

### 背景

v8 采用方案 A：不继续调 v7 的小碎片阈值，而是新增两个目标化后处理。

- 第一目标：SMV/portal 显式桥接，把分离的 portal 主组件和 SMV/脾静脉相关组件连成同一 portal 树。
- 第二目标：TotalSeg anchor 注入后再审核外周大团块，处理 `confidence=1.0` 的注入后假阳性。

### v8 实现内容

- `config/ct0021_v8.yaml`
  - 保留 v6/v7 的肝门保护、SMV/portal 保护、portal 重标、肝表面、isolated blob 与 apex/surface morph cleanup。
  - 关闭 `outer_peripheral_blob_cleanup`，避免继续把小碎片清理作为外周大团块清理。
  - 新增 `smv_portal_bridge_repair`：
    - `enabled: true`
    - `max_gap_mm: 30`
    - `corridor_radius_mm: 5`
    - `endpoint_min_volume_mm3: 300`
    - `min_portal_minus_venous_hu: 10`
    - `fallback_centerline_enabled: true`
    - `bridge_confidence: 0.85`
  - 新增 `post_anchor_peripheral_component_audit`：
    - `enabled: true`
    - `organ_envelope_dilation_mm: 18`
    - `core_anchor_protection_mm: 6`
    - `min_component_volume_mm3: 96`
    - `max_component_linearity: 3.0`
    - `confidence_max: 1.01`

- `src/ct_eus_vessel/postprocess.py`
  - 新增 `apply_smv_portal_bridge_repair`：
    - 在 portal 组件之间寻找小于 `max_gap_mm` 的候选组件对。
    - corridor 限制在 body 内、非 hard exclusion、SMV/portal protection 内。
    - 优先使用已有 venous 候选或满足 portal-venous HU margin 的空体素。
    - fallback centerline 只允许整条中心线都在 allowed 区域内时启用。
    - 为避免真实病例组件对爆炸，桥接只检查体积最大的前 16 个 endpoint 组件，并用 KDTree 与局部 bbox 连通性验证。
  - 新增 `apply_post_anchor_peripheral_component_audit`：
    - 在 TotalSeg anchor 注入后运行。
    - 删除 organ envelope 外、核心 anchor/bridge 保护外、体积达到阈值且非管状的大组件。
    - 不因 `confidence=1.0` 自动保护组件，专门处理注入后的高置信假阳性。

- `src/ct_eus_vessel/pipeline.py`
  - 在 `inject_totalseg_vessel_anchors` 后执行 `smv_portal_bridge_repair`。
  - 再执行 `post_anchor_peripheral_component_audit`，最后才进入 table-like component gate。
  - post-anchor audit 的核心保护只包括靠近 organ envelope 的 TotalSeg vessel anchors 和本轮新增 bridge voxels；不再把整片 SMV protection 或所有 portal 标签都保护起来。
  - 新增质量指标：
    - `smv_portal_bridge_repair_voxels`
    - `smv_portal_bridge_repair_pairs`
    - `smv_portal_bridge_repair_fallback_voxels`
    - `smv_portal_bridge_repair_max_gap_mm`
    - `post_anchor_peripheral_cleanup_voxels`
    - `post_anchor_peripheral_cleanup_components`
    - `post_anchor_peripheral_cleanup_by_label`

### v8 测试

- 新增配置测试：
  - `ct0021_v8.yaml` 关闭 v7 的 `outer_peripheral_blob_cleanup`，启用 SMV bridge repair 与 post-anchor audit。
- 新增后处理纯函数测试：
  - SMV/portal 两个大组件经 venous/HU corridor 桥接为 portal。
  - fallback centerline 只在 body、非 hard、protection 内启用；hard exclusion 截断时不桥接。
  - post-anchor audit 可删除 `confidence=1.0` 的外周大 portal anchor 团块，并保留核心保护区和高线性分支。
- 新增 pipeline 集成测试：
  - 模拟 TotalSeg 注入后外周 portal 团块，v8 在注入后删除。
  - 模拟两个 portal 组件中间存在 venous bridge candidate，v8 将 bridge candidate 改为 portal。
- 自动化验证：
  - `PYTHONPATH=src pytest tests/test_config.py tests/test_postprocess.py tests/test_pipeline_auto.py -q`：`35 passed, 6 warnings`
  - `PYTHONPATH=src pytest -q`：`81 passed, 12 warnings`

### v8 真实病例复跑状态

- 2026-06-15 已完成真实 CT-0021 复跑，覆盖主 v8 输出目录：
  - `/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v8-post-anchor-audit-smv-bridge`
- 运行命令：
  - `PYTHONPATH=src python -m ct_eus_vessel.cli run --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' --output '/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v8-post-anchor-audit-smv-bridge' --config config/ct0021_v8.yaml --vesselness-mode slice-frangi`
- 输出完整性：
  - 已生成 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`、`vessel_confidence.nrrd`、`mesh/vessel_fused_slicer_ras.ply`、`metrics_report.json`、`compare_report.json`。
  - 根目录无 `.nii.gz`；兼容 NIfTI 均在 `compat_nifti/`。
  - `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd` 的 size、spacing、origin、direction 一致。
  - `mesh_written: true`，`warnings: []`。
- v8 真实病例核心质量指标：
  - `voxel_counts`: arterial `89522`，portal `18963`，venous `349103`，fused `612849`。
  - `smv_portal_bridge_repair_voxels: 3267`
  - `smv_portal_bridge_repair_pairs: 1`
  - `smv_portal_bridge_repair_fallback_voxels: 17`
  - `smv_portal_bridge_repair_max_gap_mm: 22.1053389784581`
  - `portal_relabel_voxels: 12616`
  - `portal_relabel_bridge_voxels: 6389`
  - `post_anchor_peripheral_cleanup_voxels: 50861`
  - `post_anchor_peripheral_cleanup_components: 8`
  - `post_anchor_peripheral_cleanup_by_label`: arterial `19956`，portal `551`，venous `30354`
  - `outside_body_voxels: 0`
- 与人工参考输出 compare：
  - overall precision `0.7529489319555062`，recall `0.8181790779607404`，dice `0.7842098912509623`
  - arterial dice `0.7955906701781289`
  - portal dice `0.5708972000817495`
  - venous dice `0.7810058042417336`
  - liver overall dice `0.6613013329500952`
- portal 连通性复测：
  - v7 portal top components: `51703, 10606, 1235, ...`，第二大组件到最大组件 gap `22.1053389784581 mm`。
  - v8 portal top components: `65567, 1235, 207, ...`，原 v7 第二大 portal 分量已并入最大分量。
  - v8 当前第二大组件到最大组件 gap `17.36221787379338 mm`，对应剩余较小分支，不再是 v7 的 SMV/portal 主断裂。
- 待人工 Slicer 复核：
  - 按项目规则加载同一输出目录根部 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`。
  - 复测前删除旧 volume、segmentation、closed surface 和 model 节点，避免旧缓存误导。
  - 重点复核用户指出的外周团块是否被 `post_anchor_peripheral_component_audit` 删除，以及 SMV/portal 主干连续性。

## CT-0021 v9 规划、实现与真实病例结果

### 背景

- 用户复核 v8 后指出：
  - SMV/portal 有一点连接，但只是稀疏连接，不像一根血管。
  - 肝脏表面蓝圈区域的片状组织仍未清干净，该问题从 v6 开始持续存在。
- v9 约束：
  - 人工标注不能参与重建，只能作为重建结果评估的 groundtruth。
  - SMV 修复必须按 CT/HU、候选 mask、TotalSeg prior 与 protection mask 的事实证据进行，不能为了视觉效果凭空补血管。

### v9 实现内容

- 新增 `config/ct0021_v9.yaml`
  - 基于 v8，保留 post-anchor audit。
  - `smv_portal_bridge_repair` 新增证据约束 tube fill：
    - `fallback_centerline_enabled: false`
    - `morphological_tube_fill_enabled: true`
    - `tube_radius_mm: 3.0`
    - `closing_radius_mm: 1.5`
    - `min_evidence_fraction: 0.35`
    - `max_fill_to_evidence_ratio: 1.75`
  - 新增 `liver_surface_sheet_cleanup`：
    - `surface_depth_mm: 15`
    - `min_component_volume_mm3: 200`
    - `max_component_volume_mm3: 12000`
    - `max_component_linearity: 4.5`
    - `min_surface_fraction: 0.55`
    - `confidence_max: 1.01`
    - `core_anchor_protection_mm: 8`
    - `bridge_protection_mm: 5`
    - `target_labels: [arterial, portal, venous]`
- `src/ct_eus_vessel/postprocess.py`
  - `apply_smv_portal_bridge_repair` 保持 v8 endpoint/pair 搜索，但在启用 tube fill 时：
    - 只在 pair 局部 bbox 内构建 line/tube/corridor，避免真实体积全图 distance transform 的内存峰值。
    - evidence 来自已有 venous candidate 或 portal/venous coverage 同时存在且 `portal_hu - venous_hu >= min_portal_minus_venous_hu` 的体素。
    - evidence fraction 或 fill/evidence ratio 不达标时拒绝桥接，并记录 rejected reason。
    - 最近点 tie 时优先选靠近两个组件质心的端点，避免桥接轴偏到组件边缘。
  - 新增 `apply_liver_surface_sheet_cleanup`：
    - 使用自动 liver mask 的 surface band、body、hard exclusion、core anchor protection、bridge protection。
    - 删除肝表面带内达到体积阈值、非高线性、且非 protected 的片状组件。
    - surface band 使用按物理轴向扫描的轻量实现，避免全 liver EDT 导致真实病例内存压力过高。
- `src/ct_eus_vessel/pipeline.py`
  - 在 post-anchor audit 后、table-like gate 前执行 `liver_surface_sheet_cleanup`。
  - 新增质量指标：
    - `smv_portal_bridge_repair_evidence_voxels`
    - `smv_portal_bridge_repair_morph_fill_voxels`
    - `smv_portal_bridge_repair_rejected_pairs`
    - `smv_portal_bridge_repair_rejected_by_reason`
    - `liver_surface_sheet_cleanup_voxels`
    - `liver_surface_sheet_cleanup_components`
    - `liver_surface_sheet_cleanup_by_label`
    - `liver_surface_sheet_cleanup_candidate_voxels`
    - `liver_surface_sheet_cleanup_protected_voxels`

### v9 测试

- 新增配置测试：
  - `ct0021_v9.yaml` 启用 SMV tube fill 与 liver surface sheet cleanup。
- 新增后处理纯函数测试：
  - 稀疏 SMV/portal bridge 在 HU/venous evidence 支持下加厚为 tube-like bridge。
  - 无 evidence 时拒绝 tube fill，并记录 `insufficient_evidence`。
  - liver surface sheet cleanup 删除片状表面组织，并保留高线性分支、core anchor 附近组件和 bridge 附近组件。
- 新增 pipeline 集成测试：
  - `label_path=None` 的自动重建路径下，V9 能 tube-fill bridge 并删除 liver surface sheet。
  - 断言 `summary["label"] is None` 与 `guidance_source == "auto_totalseg_priors"`。
- 自动化验证：
  - `PYTHONPATH=src pytest -q`：`86 passed, 12 warnings`

### v9 真实病例复跑状态

- 2026-06-15 已完成真实 CT-0021 复跑，输出目录：
  - `/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v9-smv-tube-surface-sheet-cleanup`
- 运行命令：
  - `PYTHONPATH=src python -m ct_eus_vessel.cli run --input '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例/CT-0021' --output '/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v9-smv-tube-surface-sheet-cleanup' --config config/ct0021_v9.yaml --vesselness-mode slice-frangi`
- 重建输入约束：
  - `metrics_report.json` 中 `label: null`
  - `guidance_source: auto_totalseg_priors`
  - 人工参考只用于后续 compare，不参与重建。
- 输出完整性：
  - 已生成 `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`、`vessel_confidence.nrrd`、`mesh/vessel_fused_slicer_ras.ply`、`metrics_report.json`、`compare_report.json`。
  - 根目录无 `.nii.gz`；兼容 NIfTI 均在 `compat_nifti/`。
  - `reference_ct.nrrd`、`vessel_fused_multilabel.nrrd`、`vessel_confidence.nrrd` 的 size、spacing、origin、direction 一致。
  - `mesh_written: true`，`warnings: []`。
- v9 真实病例核心质量指标：
  - `voxel_counts`: arterial `89522`，portal `18963`，venous `349103`，fused `574032`
  - `smv_portal_bridge_repair_voxels: 1854`
  - `smv_portal_bridge_repair_pairs: 1`
  - `smv_portal_bridge_repair_evidence_voxels: 1113`
  - `smv_portal_bridge_repair_morph_fill_voxels: 1854`
  - `smv_portal_bridge_repair_rejected_pairs: 0`
  - `liver_surface_sheet_cleanup_voxels: 37404`
  - `liver_surface_sheet_cleanup_components: 3`
  - `liver_surface_sheet_cleanup_by_label`: arterial `0`，portal `1157`，venous `36247`
  - `post_anchor_peripheral_cleanup_voxels: 50861`
  - `outside_body_voxels: 0`
- 与人工参考输出 compare：
  - overall precision `0.7560588956713215`，recall `0.7695221006083452`，dice `0.7627310919570025`
  - arterial dice `0.7955906701781289`
  - portal precision `0.5849271530629465`，portal dice `0.5767993343502982`
  - venous dice `0.7491311878730376`
  - liver overall dice `0.5206459151638018`
- v8 对比解读：
  - v9 fused voxels 从 v8 `612849` 降至 `574032`，主要来自新增 liver surface sheet cleanup。
  - portal dice 从 v8 `0.5708972000817495` 小幅升至 v9 `0.5767993343502982`，portal precision 从 `0.5687043622248161` 升至 `0.5849271530629465`。
  - overall dice 从 v8 `0.7842098912509623` 降至 v9 `0.7627310919570025`，liver overall dice 从 `0.6613013329500952` 降至 `0.5206459151638018`，提示 surface sheet cleanup 可能牺牲一部分人工参考中的肝内/肝表面召回，需要视觉复核确认是否是期望删除。
- portal 连通性复测：
  - v7 portal top components: `51703, 10606, 1235, ...`，第二大组件到最大组件 gap `22.1053389784581 mm`。
  - v8 portal top components: `65567, 1235, 207, ...`，原 v7 第二大 portal 分量已并入最大分量。
  - v9 portal top components: `63560, 1235, 196, ...`，原 v7 第二大 portal 分量未重新断开。
  - v9 portal components 从 v8 `1037` 降至 `960`，portal voxels 从 v8 `73678` 降至 `71108`。
- 待人工 Slicer 复核：
  - 按项目规则加载同一输出目录根部 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`。
  - 复测前删除旧 volume、segmentation、closed surface 和 model 节点，避免旧缓存误导。
  - 重点复核：
    - SMV/portal 蓝圈区域是否从 v8 稀疏丝状连接改善为更连续、且不越界的管状连接。
    - 肝表面蓝圈区域的片状组织是否被清理。
    - liver/venous recall 下降是否对应误删真实肝内血管。

## CT-0021 v10 回退基线

### 背景

- v9 的 SMV/portal bridge 已经稳定，用户随后希望优先“保住肝内大支杆”，同时适当放松肝尖清理。
- v10 复盘结论：
  - SMV 重建保持良好，不应被后续更激进的 apex 保护逻辑破坏。
  - 当前仍剩两个问题：
    - 肝尖表层下方约 1cm 的片状/网状残留没有剔除干净。
    - 肝内大血管与主干之间有断裂/切断问题。
- 因此 v10 作为后续迭代的回退基线，先保住 v9 的稳定 SMV/portal，再逐步处理肝尖深层残留。

### v10 回退内容

- 新增 `config/ct0021_v10.yaml`
  - 以 v9 为基础，保持 `smv_portal_bridge_repair`、`liver_surface_sheet_cleanup` 与 post-anchor audit。
  - 保持 `bridge_protection_mm: 5`，不引入 v11 的 `bridge_protection_mm: 12`。
  - 不启用 `apex_subsurface_cleanup`。
- `src/ct_eus_vessel/pipeline.py`
  - v10 路径只走 v9 的 apex / liver surface 清理，不进入 v11 的 subsurface 清理分支。
- `tests/test_config.py`
  - 增加 v10 配置回归，确认 `apex_subsurface_cleanup` 未启用。
- `tests/test_pipeline_auto.py`
  - 增加 v10 pipeline 回归，确认 v10 基线不会调用 `apply_apex_subsurface_cleanup`。

### v10 真实病例结果

- 真实输出目录：
  - `/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v10-branch-preserve-relaxed-apex`
- 核心指标：
  - `voxel_counts`: arterial `89522`，portal `18963`，venous `349103`，fused `599540`
  - `smv_portal_bridge_repair_voxels: 1854`
  - `smv_portal_bridge_repair_pairs: 1`
  - `smv_portal_bridge_repair_evidence_voxels: 1113`
  - `smv_portal_bridge_repair_morph_fill_voxels: 1854`
  - `portal_relabel_voxels: 12616`
  - `portal_relabel_bridge_voxels: 6389`
  - `final_liver_surface_cleanup_voxels: 30320`
  - `apex_surface_cleanup_voxels: 5905`
  - `liver_surface_sheet_cleanup_voxels: 11896`
  - `liver_surface_sheet_cleanup_components: 3`
  - `liver_surface_sheet_cleanup_by_label`: arterial `0`，portal `956`，venous `10940`
  - `outside_body_voxels: 0`
  - `voxels_removed_by_table_gate: 0`

### v10 现状结论

- v10 的 SMV/portal 重建是稳定的，适合作为回退基线。
- v10 仍保留用户指出的两个问题：
  - 肝尖表层下的片状/网状残留。
  - 肝内大血管与主干的连续性问题。
- 后续迭代应围绕这两个问题单独处理，不要再把 v10 的主干稳定性打散。

## CT-0021 v11 规划与实现

### 背景

- 用户把目标收敛为：
  - 优先保住肝内大支杆。
  - 适当放松肝尖清理，但要优先处理肝尖表面下方一定深度的片状/网状残留。
- v10 复盘后的关键判断：
  - 肝内大血管“回来但断裂”更像 bridge repair 和保护区顺序问题，不是单纯阈值不足。
  - 肝尖表面下的残留不是“找不到位置”，而是现有清理只覆盖到表面层，深一层的片状组织还没有单独的清理通道。

### v11 目标

- 保住肝内大支杆。
- 在不误伤 trunk 的前提下，清理肝尖表面下的片状/网状组织。

### v11 实现内容

- 新增 `config/ct0021_v11.yaml`
  - 基于 v9，保留 `smv_portal_bridge_repair`、`liver_surface_sheet_cleanup` 和 post-anchor audit。
  - `liver_surface_sheet_cleanup.bridge_protection_mm` 调到 `12`，优先保住肝内大支杆与 SMV 连续性。
  - 新增 `apex_subsurface_cleanup`：
    - `enabled: true`
    - `apex_fraction: 0.12`
    - `subsurface_min_depth_mm: 3`
    - `subsurface_max_depth_mm: 8`
    - `confidence_min: 0.80`
    - `min_component_volume_mm3: 120`
    - `max_component_volume_mm3: 1500`
    - `max_component_linearity: 4.5`
    - `min_surface_fraction: 0.35`
    - `anchor_dilation_mm: 1`
- `src/ct_eus_vessel/postprocess.py`
  - `apply_apex_surface_morph_cleanup` 的 anchor 保护区略收紧，避免把贴着 trunk 的一层表面片块一起护住。
  - `apply_apex_subsurface_cleanup` 专门处理肝尖表面下方一定深度的 venous 片状组织，并保留 protected trunk。
- `src/ct_eus_vessel/pipeline.py`
  - 保留 SMV bridge repair 作为前置主干修复。
  - 在 apex / liver surface 系列清理前构建 `protected_trunk_mask`，用于保护桥接后的主干连续性。
  - 新增质量指标：
    - `apex_subsurface_cleanup_voxels`
    - `apex_subsurface_cleanup_components`
    - `apex_subsurface_cleanup_by_label`
    - `apex_subsurface_cleanup_by_region`
    - `apex_subsurface_cleanup_candidate_voxels`
    - `apex_subsurface_cleanup_protected_voxels`

### v11 测试

- 新增配置测试：
  - `ct0021_v11.yaml` 启用 `apex_subsurface_cleanup`。
- 新增后处理纯函数测试：
  - `apex_surface_morph_cleanup` 保留 protected trunk，删除 apex 表面片块。
  - `apex_subsurface_cleanup` 删除更深层的肝尖片块，保留 trunk。
- 新增 pipeline 集成测试：
  - `label_path=None` 路径下，SMV/portal bridge 仍能接上。
  - 肝尖表面下残留能被 `apex_subsurface_cleanup` 清理。

## CT-0021 v12b Git 基线

### 背景

- v12b 被设为当前代码迭代的 Git 初始化基线，方便后续小步试验、回退和对比。
- 用户复核 v12b 后确认仍存在两个未解决问题：
  - 肝内主血管已经回来，但没有和主干连接上，表现为断裂/切断。
  - 肝尖浅表层组织仍未清理干净，重点位置是肝尖肝表面下方约 1cm 的片状/破洞网状组织。
- v12b 的 SMV/portal bridge 指标恢复到历史较好水平，但该指标只能说明 SMV/portal 局部桥接存在，不能证明肝内主血管与主干已经连续。

### v12b 真实病例结果

- 真实输出目录：
  - `/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/无标签/CT-0021-v12b-bridge-seed-sheet-protect`
- 核心指标：
  - `voxel_counts`: arterial `89522`，portal `18963`，venous `349064`，fused `592428`
  - `smv_portal_bridge_repair_voxels: 1854`
  - `smv_portal_bridge_repair_pairs: 1`
  - `smv_portal_bridge_repair_evidence_voxels: 1113`
  - `smv_portal_bridge_repair_morph_fill_voxels: 1854`
  - `portal_relabel_voxels: 12616`
  - `portal_relabel_bridge_voxels: 6389`
  - `final_liver_surface_cleanup_voxels: 30318`
  - `apex_surface_cleanup_voxels: 8643`
  - `apex_subsurface_cleanup_voxels: 0`
  - `liver_surface_sheet_cleanup_voxels: 16256`
  - `liver_surface_sheet_cleanup_components: 4`
  - `liver_surface_sheet_cleanup_by_label`: arterial `0`，portal `0`，venous `16256`
  - `post_anchor_peripheral_cleanup_voxels: 50861`
  - `protected_trunk_voxels: 133112`
  - `outside_body_voxels: 0`
  - `voxels_removed_by_table_gate: 0`
  - `warnings: []`

### 下一轮约束

- 以 v12b Git tag 作为可回退基线，不再让 SMV/portal bridge 回退到 v11 的消失状态。
- 新增或修改清理逻辑前，必须同时检查：
  - SMV/portal bridge repair 是否仍有 `1854` 左右的证据约束补全体素。
  - 肝内主血管与主干是否在三维连通性上属于同一组件，不能只看局部 bridge 指标。
  - 肝尖浅表下方清理是否覆盖到 apex 上方更大比例的 liver mask，而不是只清理最尖端 `apex_fraction: 0.12` 的表面层。
- 后续 Slicer 复核仍按项目规则加载同一输出目录根部 `reference_ct.nrrd` 与 `vessel_fused_multilabel.nrrd`，或直接加载 `mesh/vessel_fused_slicer_ras.ply`；复测前删除旧 volume、segmentation、closed surface 和 model 节点。
