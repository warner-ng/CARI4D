# CARI4D Bike 全链路复盘（Date03_Sub01_bike_on_may_29_21_17）

## 1. 目标

- 序列：
  - `SEQ_RAW=bike_on_may_29_21_17`
  - `SEQ=Date03_Sub01_bike_on_may_29_21_17`
- 目标：
  1. 跑通 CARI4D 从分割到 Step7 优化。
  2. 解决纹理不一致与 Step6 OOM。

---

## 2. 关键输入/路径

- 输入视频：`/home/warner/_projects/CARI4D/bike_on_may_29_21_17.mov`
- videogen 根目录：`data/cari4d-demo/videogen`
- 主视频产物：`data/cari4d-demo/videogen/videos/Date03_Sub01_bike_on_may_29_21_17.0.color.mp4`

---

## 3. 各步骤产物

## Step1：SAM3 分割

产物：
- `data/bike_on_may_29_21_17/masks/bike_on_may_29_21_17_masks_k0.h5`
- `data/bike_on_may_29_21_17/masks/bike_on_may_29_21_17_sam3_vis.mp4`

映射到 videogen：
- `data/cari4d-demo/videogen/masks/Date03_Sub01_bike_on_may_29_21_17_masks_k0.h5`（软链）
- H5 group 同时包含：
  - `bike_on_may_29_21_17`
  - `Date03_Sub01_bike_on_may_29_21_17`

## Step2：视频标准化（ffmpeg）

- `data/cari4d-demo/videogen/videos/Date03_Sub01_bike_on_may_29_21_17.0.color.mp4`（240帧）

## Step3：HY3D 重建 + Blender 转 OBJ

raw 目录产物：
- `data/bike_on_may_29_21_17/meshes/bike_on_may_29_21_17_000_rgba/bike_on_may_29_21_17_000_rgba.png`
- `.../bike_on_may_29_21_17_000_rgba.glb`
- `.../bike_on_may_29_21_17_000_align.obj`

映射/导出：
- `data/cari4d-demo/videogen/meshes -> data/bike_on_may_29_21_17/meshes`（软链）
- `data/cari4d-demo/videogen/meshes/Date03_Sub01_bike_on_may_29_21_17_000_rgba/Date03_Sub01_bike_on_may_29_21_17_000_align.obj`

## Step4：Sapiens 2D 关键点

- `data/cari4d-demo/videogen/packed/Date03_Sub01_bike_on_may_29_21_17_GT-packed.pkl`

## Step5：demo-custom 前半段（Depth/NLF/Scale/FP）

UniDepth + NLF：
- `data/cari4d-demo/videogen/videos/Date03_Sub01_bike_on_may_29_21_17.0.depth-reg.mp4`
- `data/cari4d-demo/videogen/nlf/Date03_Sub01_bike_on_may_29_21_17_params.pkl`
- `data/cari4d-demo/videogen/nlf-opt/Date03_Sub01_bike_on_may_29_21_17_params.pkl`
- `data/cari4d-demo/videogen/videos-aligned/Date03_Sub01_bike_on_may_29_21_17.0.color.mp4`
- `data/cari4d-demo/videogen/videos-aligned/Date03_Sub01_bike_on_may_29_21_17.0.depth-reg_aligned.pkl`

尺度估计：
- `data/cari4d-demo/videogen/meshes-metric/Date03_Sub01_bike_on_may_29_21_17_000_align/`
  - `..._align.obj`
  - `..._align_align.obj`
  - `..._fp-res.json`
  - `..._fp-res-refine.json`

FP 跟踪：
- `data/cari4d-demo/videogen/fp-hy3d3-track/Date03_Sub01_bike_on_may_29_21_17_all_k0.pkl`
- `data/cari4d-demo/videogen/fp-hy3d3-track/Date03_Sub01_bike_on_may_29_21_17_all.pkl`
- `data/cari4d-demo/videogen/fp-hy3d3-track/Date03_Sub01_bike_on_may_29_21_17_all_k0_k0.mp4`

## Step6：CoCoNet（已跑通）

- `output/coconet/cari4d-release+step031397_demo_oomfix/Date03_Sub01_bike_on_may_29_21_17.pth`
- `output/viz/cari4d-release+step031397_demo_oomfix+Date03_Sub01_bike_on_may_29_21_17_it1_input.mp4`

## Step7：优化（已跑通）

- `output/opt/cari4d-release+step031397_demo_oomfix-hy3d3-optv2_oomfix/Date03_Sub01_bike_on_may_29_21_17.pth`
- 阶段可视化：
  - `...+step000000.mp4`
  - `...+step000500.mp4`
  - `...+step001000.mp4`
  - `...+step001500.mp4`
  - `...+step002000.mp4`
  - `...+step002500.mp4`
  - `...+step003000.mp4`

---

## 4. 关键问题与修复

### 问题 A：纹理不一致（join_meshes_as_batch）

现象：
- `ValueError: Inconsistent textures in join_meshes_as_batch.`

修复：
- 对 metric mesh 做最小纹理补齐（`mtllib/map_Kd/vt/f v/v`）。
- 对无 UV 场景使用 vertex color fallback，避免模板拼接失败。

### 问题 B：Step6 OOM

现象：
- `run_horefine.py` 在 `render_front_side` 阶段 OOM（约 5.47 GiB 分配失败）。

修复参数：
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`
- `clip_len=8`（必要时可降到 `4`）

---

## 5. 结论

- 本序列已完成从 Step1 到 Step7 的全链路。
- 后续复用建议：
  1. Step6 默认启用 `clip_len=8`。
  2. 显存碎片场景优先设 `expandable_segments + max_split_size_mb`。
  3. 优先单步重跑 Step6/Step7，避免全流程重复耗时。
