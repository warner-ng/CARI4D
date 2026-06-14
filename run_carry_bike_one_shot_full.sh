#!/usr/bin/env bash
set -euo pipefail

cd /home/warner/_projects/CARI4D

# total time: 50 min for all


# Outputs for the current SEQ:
# - Masks + SAM3 visualization: data/${SEQ_RAW}/masks/
# - Normalized input video/depth/aligned video: data/cari4d-demo/videogen/videos*/${SEQ}.*
# - HY3D raw mesh: data/${SEQ_RAW}/meshes/
# - HY3D aliases/metric mesh: data/cari4d-demo/videogen/meshes*/${SEQ}_000_*/
# - Sapiens packed pose: data/cari4d-demo/videogen/packed/${SEQ}_GT-packed.pkl
# - NLF/SMPLH visualizations: data/cari4d-demo/videogen/nlf*/${SEQ}_params*.mp4
# - FoundationPose visualization/cache: data/cari4d-demo/videogen/fp-hy3d3-track/${SEQ}_*
# - CoCoNet result: output/coconet/cari4d-release+step031397_demo_*/${SEQ}.pth
# - Final optimization videos/result: output/opt/*/${SEQ}+step*.mp4 and output/opt/*/${SEQ}.pth

HUMAN_PROMPT="person"  # CHANGE_ME: 每次换新视频时，按视频里的人改 prompt
OBJECT_PROMPT="bike"  # CHANGE_ME: 每次换新视频时，按交互物体改 prompt

SEQ_RAW="bike_LA"  # CHANGE_ME: 每次换新视频时，改成短名字
SEQ="Date03_Sub01_bike_LA"  # CHANGE_ME: 每次换新视频时，改成完整序列名
VIDEO_IN="/home/warner/_projects/CARI4D/bike_LA.MP4"  # CHANGE_ME: 每次换新视频时，改成源视频路径
BLENDER_PATH="/home/warner/tools/blender-3.6.17-linux-x64/blender"  # CHANGE_ME: 只有 Blender 路径变了才改
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # CHANGE_ME: 已有产物默认跳过；设为 0 强制重跑

DATA_SEQ_ROOT="data/${SEQ_RAW}"
MASKS_ROOT="${DATA_SEQ_ROOT}/masks"
MESHES_ROOT="${DATA_SEQ_ROOT}/meshes"
VIDEOGEN_ROOT="data/cari4d-demo/videogen"
VIDEO_OUT="${VIDEOGEN_ROOT}/videos/${SEQ}.0.color.mp4"
MASKS_RAW="${MASKS_ROOT}/${SEQ_RAW}_masks_k0.h5"
MASKS_IN="${MASKS_ROOT}/${SEQ}_masks_k0.h5"
MASKS_OUT="${VIDEOGEN_ROOT}/masks/${SEQ}_masks_k0.h5"
HY3D_RAW_OBJ="${MESHES_ROOT}/${SEQ_RAW}_000_rgba/${SEQ_RAW}_000_align.obj"
HY3D_SEQ_OBJ="${VIDEOGEN_ROOT}/meshes/${SEQ}_000_rgba/${SEQ}_000_rgba.obj"
SAPIENS_PACKED="${VIDEOGEN_ROOT}/packed/${SEQ}_GT-packed.pkl"

run_step() {
  echo
  echo "========== $* =========="
}

run_step "check inputs"
test -f "$VIDEO_IN"
test -x "$BLENDER_PATH"
mkdir -p \
  "${VIDEOGEN_ROOT}/videos" "${VIDEOGEN_ROOT}/masks" "${VIDEOGEN_ROOT}/meshes-metric" "${VIDEOGEN_ROOT}/fp-hy3d3-track" "${VIDEOGEN_ROOT}/packed" \
  "${DATA_SEQ_ROOT}/videos" "${MASKS_ROOT}" "${MESHES_ROOT}"

run_step "step 1 masks with SAM3"
source /home/warner/miniconda3/etc/profile.d/conda.sh
set +u
conda activate sam3
set -u
if [[ "$SKIP_EXISTING" == "1" && -f "$MASKS_RAW" ]]; then
  echo "skip SAM3 masks: $MASKS_RAW"
else
  HF_TOKEN="$(hf auth token)"
  export HF_TOKEN
  python prep/run_sam3_masks.py \
    --video "$VIDEO_IN" \
    --human_prompt "$HUMAN_PROMPT" \
    --object_prompt "$OBJECT_PROMPT" \
    --output_dir "$MASKS_ROOT" \
    --visualize
fi
test -f "$MASKS_RAW"
ln -sfn "$(realpath "$MASKS_RAW")" "$MASKS_IN"
test -f "$MASKS_IN"

run_step "step 1.5 normalize video and masks"
if [[ "$SKIP_EXISTING" == "1" && -f "$VIDEO_OUT" ]]; then
  echo "skip normalized video: $VIDEO_OUT"
else
  rm -f "$VIDEO_OUT"
  ffmpeg -y -i "$VIDEO_IN" -pix_fmt yuv420p "$VIDEO_OUT"
fi
ln -sfn "$(realpath "$MASKS_ROOT")" "${VIDEOGEN_ROOT}/masks"
ln -sfn "$(realpath "$MASKS_IN")" "$MASKS_OUT"

python - <<PY
import h5py

p = "${MASKS_OUT}"
aliases = ["${SEQ_RAW}", "${SEQ}"]

with h5py.File(p, "r+") as f:
    existing = list(f.keys())
    if not existing:
        raise RuntimeError(f"No groups found in {p}")

    src = None
    for name in aliases:
        if name in f:
            src = name
            break
    if src is None:
        src = existing[0]

    for name in aliases:
        if name and name not in f:
            f[name] = f[src]

    print("mask groups:", list(f.keys()))
    print("mask source:", src)
PY

run_step "step 2 HY3D reconstruction"
set +u
conda activate hy3d
set -u
TORCH_LIB="$(python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
if [[ "$SKIP_EXISTING" == "1" && -f "$HY3D_RAW_OBJ" ]]; then
  echo "skip HY3D reconstruction: $HY3D_RAW_OBJ"
else
  python -c "import custom_rasterizer, custom_rasterizer_kernel; print('rasterizer ok')"
  python prep/run_hy3d_recon.py \
    --video "$VIDEO_IN" \
    --masks_root "$MASKS_ROOT" \
    --hy3d_root "$MESHES_ROOT" \
    --blender_path "$BLENDER_PATH"
fi

run_step "step 2.5 mesh aliases"
ln -sfn "$(realpath "$MESHES_ROOT")" "${VIDEOGEN_ROOT}/meshes"
cd "${VIDEOGEN_ROOT}/meshes"
ln -sfn "${SEQ_RAW}_000_rgba" "${SEQ}_000_rgba"
cd /home/warner/_projects/CARI4D

OUT="${VIDEOGEN_ROOT}/meshes/${SEQ}_000_rgba"
SRC_GLB="${MESHES_ROOT}/${SEQ_RAW}_000_rgba/${SEQ_RAW}_000_rgba.glb"
if [[ "$SKIP_EXISTING" == "1" && -f "$HY3D_SEQ_OBJ" ]]; then
  echo "skip sequence mesh export: $HY3D_SEQ_OBJ"
else
  rm -rf "$OUT"
  mkdir -p "$OUT"
  "$BLENDER_PATH" -b --python-expr "
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath='$SRC_GLB')
bpy.ops.wm.obj_export(filepath='$OUT/${SEQ}_000_rgba.obj', export_materials=True)
"
fi
ln -sfn "${SEQ}_000_rgba.obj" "$OUT/${SEQ}_000_align.obj"

run_step "step 3 Sapiens pose"
set +u
conda activate cari4d
set -u
unset LD_LIBRARY_PATH
if [[ "$SKIP_EXISTING" == "1" && -f "$SAPIENS_PACKED" ]]; then
  echo "skip Sapiens packed pose: $SAPIENS_PACKED"
else
  conda run -n cari4d python prep/run_sapiens_pose.py \
    --video "$VIDEO_OUT" \
    --masks_root "${VIDEOGEN_ROOT}/masks" \
    --packed_root "${VIDEOGEN_ROOT}/packed"
fi

run_step "step 4-7 docker pipeline"
docker start cari4d >/dev/null
docker exec \
  -e SEQ="$SEQ" \
  -e VIDEO_OUT="$VIDEO_OUT" \
  -e SKIP_EXISTING="$SKIP_EXISTING" \
  -e PIPELINE_PROFILE=behave_like_demo \
  -e COCONET_USE_INTERMEDIATE=False \
  -e OPT_REFINE_BS=32 \
  -e OPT_REFINE_PEN=2.0 \
  -e OPT_W_J2D=0.006 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  cari4d bash -lc '
set -euo pipefail
cd /home/warner/_projects/CARI4D
	python -c "from smplfitter.pt import BodyModel, BodyFitter; print(\"smplfitter ok\")"
	test -f "$VIDEO_OUT"
	mkdir -p .tmp_run_carry_pythonpath
	cat > .tmp_run_carry_pythonpath/sitecustomize.py <<'"'"'PY'"'"'
import torch

torch.backends.cuda.preferred_linalg_library("magma")
PY
	export PYTHONPATH="/home/warner/_projects/CARI4D/.tmp_run_carry_pythonpath:${PYTHONPATH:-}"

	env SEQ_ENV="$SEQ" python - <<'"'"'PY'"'"'
from pathlib import Path
from PIL import Image
import numpy as np, os
seq = os.environ["SEQ_ENV"]
obj = Path(f"data/cari4d-demo/videogen/meshes-metric/{seq}_000_align/{seq}_000_align.obj")
if not obj.exists(): raise SystemExit(0)
mtl = obj.with_suffix(".mtl")
png = obj.with_name(obj.stem + ".png")
lines = obj.read_text(errors="ignore").splitlines()
V,F,R=[],[],[]
for s in lines:
    if s.startswith("v "): R.append(s); p=s.split(); V.append([float(p[1]),float(p[2]),float(p[3])])
    elif s.startswith("f "):
        ids=[int((t.split("/")[0] if "/" in t else t)) for t in s.split()[1:] if (t.split("/")[0] if "/" in t else t)]
        for i in range(1,len(ids)-1): F.append([ids[0],ids[i],ids[i+1]])
if not V or not F: raise SystemExit(0)
xy=np.asarray(V)[:,:2]; mn,mx=xy.min(0),xy.max(0); uv=np.clip((xy-mn)/(mx-mn+1e-8),0,1); uv[:,1]=1-uv[:,1]
with obj.open("w") as f:
    f.write(f"mtllib {mtl.name}\nusemtl material_0\n\n")
    for s in R: f.write(s+"\n")
    for u,v in uv: f.write(f"vt {u:.6f} {v:.6f}\n")
    for a,b,c in F: f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
mtl.write_text(f"newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 2\nmap_Kd {png.name}\n")
Image.new("RGB",(4,4),(200,200,200)).save(png)
print("texture patched:", obj)
PY

if [[ "$SKIP_EXISTING" == "1" ]] && find output/opt -path "*${SEQ}.pth" -print -quit 2>/dev/null | grep -q .; then
  echo "skip docker demo-custom: final opt result already exists for ${SEQ}"
else
  bash scripts/demo-custom.sh "$VIDEO_OUT"
fi
'

echo
echo "done"
echo "video: ${VIDEO_OUT}"
echo "CoCoNet outputs: output/coconet"
echo "Opt outputs: output/opt"
