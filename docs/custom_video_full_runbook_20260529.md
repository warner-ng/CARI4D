# CARI4D 自定义视频全链路复盘（2026-05-29）

本文是我们为了解决“自定义视频从预处理到最终优化结果”而整理的实操文档，记录本次已跑通链路、关键路径约定、常见报错与修复方式。目标是让后续同学按本文一次走通，不再重复踩坑。

---

## 1. 目标与范围

- 输入：自定义视频（本次样例为 `carrying_bike.mov`）
- 目标：跑通 CARI4D 自定义视频流程，产出
  - `output/coconet/.../*.pth`
  - `output/opt/.../*.pth`
  - 对应优化可视化视频 `+step*.mp4`
- 原则：
  1. 优先通过**路径/命名/环境兼容**修复，不乱改核心算法逻辑。
  2. 每步结果可核验，失败时只从失败步继续。

---

## 2. 已跑通的高层链路（本次）

1. 生成 Step1 掩码（masks）
2. 生成 Step2 物体网格（HY3D，OBJ）
3. 生成 Step3 2D关键点（packed pkl）
4. 执行 Step4 主流程（`demo-custom.sh`），依次完成：
   - UniDepth
   - NLF
   - SMPLH 全局拟合
   - 单目深度对齐
   - 物体尺度估计
   - FoundationPose 跟踪
   - CoCoNet
   - 最终优化（opt）

本次最终成功序列名：`carry_Sub01_obj_wild001`

---

## 3. 按时间顺序的关键修复（含根因）

### T1：命名不符合脚本假设

**现象**
- `run_nlf_sepK.py` 报 `KeyError: 'bike'`

**根因**
- 脚本默认 `video_prefix.split('_')[1]` 是 `SubXX`（用于 `_sub_gender` 查询）

**修复**
- 将序列命名改为 BEHAVE 风格：`carry_Sub01_obj_wild001`
- 相关 masks / packed / meshes 路径均按同名补齐（软链接或重命名）

---

### T2：Sapiens / OpenMMLab 依赖冲突

**现象**
- `MMCV==2.2.0 incompatible`、`xtcocotools` ABI 错误

**修复**
- 回滚并固定关键版本：
  - `numpy==1.26.4`
  - `platformdirs==4.3.8`
  - `scipy==1.15.3`
- `xtcocotools` 安装采用不拉依赖方式：

```bash
python -m pip install --no-deps --no-cache-dir --no-build-isolation xtcocotools==1.14.3
```

---

### T3：`smplfitter` 与当前 torch 的 `nn.Buffer` 不兼容

**现象**
- `AttributeError: module 'torch.nn' has no attribute 'Buffer'`

**修复**
- 使用运行时兼容层（不改 torch 主版本）：
  - 在 `/tmp/sitecustomize.py` 注入兼容定义
  - 运行 Step4 相关命令前带 `PYTHONPATH=/tmp`

---

### T4：`align_monod2hum.py` OOM / 后续 `videos-aligned` 缺失

**现象**
- `nvdiffrast` 渲染阶段 OOM，导致后续找不到 `videos-aligned/*.color.mp4`

**修复**
- 降低对齐阶段显存压力并单独重跑该步骤
- 成功产出：
  - `data/carry_bike/videos-aligned/carry_Sub01_obj_wild001.0.color.mp4`
  - `data/carry_bike/videos-aligned/carry_Sub01_obj_wild001.0.depth-reg.mp4`

---

### T5：`fp_hy3d_track.py` 找不到模板 OBJ

**现象**
- `no aligned hy3d template found for carry_Sub01_obj_wild001`

**根因**
- `glob` 规则要求模板文件名/目录前缀与 `video_prefix` 一致

**修复**
- 在 `meshes-metric` 下补齐同名前缀的软链接：
  - `<seq>_000_rgba/<seq>_000_align.obj`

---

### T6：`run_horefine.py` 纹理批处理不一致

**现象**
- `ValueError: Inconsistent textures in join_meshes_as_batch`

**根因**
- OBJ 内 `mtllib` 仍指向 `Date03_*` 文件名，而当前目录只有 `<seq>_*`

**修复**
- 同目录同时补齐两套命名链接：
  - `<seq>_000_align.mtl/png`
  - `Date03_*_align.mtl/png`

---

## 4. 本次已验证成功的关键产物

- NLF优化：
  - `data/cari4d-demo/videogen/nlf-opt/carry_Sub01_obj_wild001_params.pkl`
- FP跟踪：
  - `data/cari4d-demo/videogen/fp-hy3d3-track/carry_Sub01_obj_wild001_all.pkl`
- CoCoNet：
  - `output/coconet/cari4d-release+step031397_demo/carry_Sub01_obj_wild001.pth`
- 最终优化：
  - `output/opt/cari4d-release+step031397_demo-hy3d3-optv2/carry_Sub01_obj_wild001.pth`
  - `output/opt/cari4d-release+step031397_demo-hy3d3-optv2/carry_Sub01_obj_wild001+step*.mp4`

---

## 5. 数值合理性检查结论（人+物体位姿）

检查文件：
- `output/opt/cari4d-release+step031397_demo-hy3d3-optv2/carry_Sub01_obj_wild001.pth`

结论：
- `pose_abs` 旋转矩阵数值健康（正交误差很小，`det(R)≈1`）
- 人物中心与物体中心距离曲线符合“逐步接近、约 3.5~4s 最接近、后续离开”
- 2D 投影诊断显示物体中心全程在画面内，不是“出画”问题

诊断文件：
- `/tmp/cari4d_debug/carry_Sub01_obj_wild001/carry_Sub01_obj_wild001_center_projection.mp4`
- `/tmp/cari4d_debug/carry_Sub01_obj_wild001/projection_summary.txt`

---

## 6. 与下游 GMR / ResMimic 的衔接

本仓库内已补充链路文档：

- `docs/cari4d_to_gmr_to_resmimic_chain.md`

包括：
- CARI4D 输出字段映射（`pr.smpl_pose / pr.smpl_t / pr.betas / pr.pose_abs`）
- 推荐中间导出格式（`*_for_downstream.npz`）
- GMR 与 ResMimic 的接口契约与检查项

---

## 7. 建议的收尾动作（可选）

1. 导出结果后清理兼容 shim：

```bash
rm -f /tmp/sitecustomize.py
```

2. 固化当前环境快照：

```bash
pip freeze > requirements.lock.session.txt
```

3. 若要长期维护，建议后续在独立分支评估升级到上游推荐 torch 版本，移除临时兼容逻辑。
