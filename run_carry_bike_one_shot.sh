#!/usr/bin/env bash
set -o pipefail
report_error(){ c="${1:-ERR}"; shift || true; echo "[ERROR][$c] ${*:-unknown error}"; }
run_or_report(){ "$@"; c=$?; [[ $c -eq 0 ]] || report_error "$c" "cmd failed: $*"; return 0; }
cd /home/warner/_projects/CARI4D

HUMAN_PROMPT="man"
OBJECT_PROMPT="bike"


SEQ_RAW="bike_on_may_29_21_17"
SEQ="Date03_Sub01_bike_on_may_29_21_17"
VIDEO_IN="/home/warner/_projects/CARI4D/bike_on_may_29_21_17.mov"
MASKS_IN="data/bike_on_may_29_21_17/masks/bike_on_may_29_21_17_masks_k0.h5"
BLENDER_PATH="/home/warner/tools/blender-3.6.17-linux-x64/blender"

DATA_SEQ_ROOT="data/${SEQ_RAW}"
MASKS_ROOT="${DATA_SEQ_ROOT}/masks"
MESHES_ROOT="${DATA_SEQ_ROOT}/meshes"

VIDEOGEN_ROOT="data/cari4d-demo/videogen"
VIDEO_OUT="${VIDEOGEN_ROOT}/videos/${SEQ}.0.color.mp4"
MASKS_OUT="${VIDEOGEN_ROOT}/masks/${SEQ}_masks_k0.h5"

[[ -n "${SEQ_RAW:-}" ]] || report_error "CHECK" "SEQ_RAW 不能为空"
[[ -n "${SEQ:-}" ]] || report_error "CHECK" "SEQ 不能为空"
[[ -n "${VIDEO_IN:-}" ]] || report_error "CHECK" "VIDEO_IN 不能为空"
[[ -n "${MASKS_IN:-}" ]] || report_error "CHECK" "MASKS_IN 不能为空"
[[ -n "${BLENDER_PATH:-}" ]] || report_error "CHECK" "BLENDER_PATH 不能为空"
[[ -n "${HUMAN_PROMPT:-}" ]] || report_error "CHECK" "HUMAN_PROMPT 不能为空"
[[ -n "${OBJECT_PROMPT:-}" ]] || report_error "CHECK" "OBJECT_PROMPT 不能为空"

run_or_report test -f "$VIDEO_IN"
mkdir -p \
  "${VIDEOGEN_ROOT}/videos" "${VIDEOGEN_ROOT}/masks" "${VIDEOGEN_ROOT}/meshes-metric" "${VIDEOGEN_ROOT}/fp-hy3d3-track" "${VIDEOGEN_ROOT}/packed" \
  "${DATA_SEQ_ROOT}/videos" "${MASKS_ROOT}" "${MESHES_ROOT}"

source /home/warner/miniconda3/etc/profile.d/conda.sh
conda activate sam3
HF_TOKEN="$(hf auth token)"
export HF_TOKEN
run_or_report python prep/run_sam3_masks.py \
  --video "$VIDEO_IN" \
  --human_prompt "$HUMAN_PROMPT" \
  --object_prompt "$OBJECT_PROMPT" \
  --output_dir "$MASKS_ROOT" \
  --visualize








run_or_report test -f "$MASKS_IN"

rm -f "$VIDEO_OUT"
ffmpeg -y -i "$VIDEO_IN" -frames:v 240 -pix_fmt yuv420p "$VIDEO_OUT"



src_m="$(realpath "$MASKS_IN")"
ln -sfn "$src_m" "$MASKS_OUT"

# 只在 group 名不一致时做一次
# Add H5 group aliases when names are inconsistent; no data copy
python - <<PY
import h5py

p = "data/cari4d-demo/videogen/masks/${SEQ}_masks_k0.h5"

aliases = ["${SEQ_RAW}", "${SEQ}"]

with h5py.File(p, "r+") as f:
    existing = list(f.keys())
    if not existing:
        raise RuntimeError(f"No groups found in {p}")

    # Prefer an existing known group as source
    src = None
    for name in aliases:
        if name in f:
            src = name
            break
    if src is None:
        src = existing[0]

    for name in aliases:
        if name and name not in f:
            f[name] = f[src]   # HDF5 hard link, no data copy

    print("groups:", list(f.keys()))
    print("source:", src)
    print("ok")
PY

echo "VIDEO_IN=$VIDEO_IN"
echo "MASKS_ROOT=$(dirname "$MASKS_IN")"
echo "HY3D_ROOT=data/${SEQ_RAW}/meshes"
echo "BLENDER_PATH=$BLENDER_PATH"
[[ -f "$VIDEO_IN" ]] && echo "video ok" || echo "video missing"
[[ -d "$(dirname "$MASKS_IN")" ]] && echo "masks_root ok" || echo "masks_root missing"
[[ -x "$BLENDER_PATH" ]] && echo "blender ok" || echo "blender missing"

source /home/warner/miniconda3/etc/profile.d/conda.sh
conda activate hy3d
TORCH_LIB=$(python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

python -c "import custom_rasterizer, custom_rasterizer_kernel; print('rasterizer ok')"

python prep/run_hy3d_recon.py \
  --video "$VIDEO_IN" \
  --masks_root "$(dirname "$MASKS_IN")" \
  --hy3d_root "data/${SEQ_RAW}/meshes" \
  --blender_path "$BLENDER_PATH"

src_mesh="$(realpath "data/${SEQ_RAW}/meshes")"
ln -sfn "$src_mesh" "${VIDEOGEN_ROOT}/meshes"

OUT="${VIDEOGEN_ROOT}/meshes/${SEQ}_000_rgba"
SRC_GLB="data/${SEQ_RAW}/meshes/${SEQ_RAW}_000_rgba/${SEQ_RAW}_000_rgba.glb"

rm -rf "$OUT"
mkdir -p "$OUT"

"$BLENDER_PATH" -b --python-expr "
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath='$SRC_GLB')
bpy.ops.wm.obj_export(filepath='$OUT/${SEQ}_000_align.obj', export_materials=True)
"








cd /home/warner/_projects/CARI4D
source /home/warner/miniconda3/etc/profile.d/conda.sh
conda activate cari4d
unset LD_LIBRARY_PATH

run_or_report conda run -n cari4d python prep/run_sapiens_pose.py \
  --video "$VIDEO_OUT" \
  --masks_root "${VIDEOGEN_ROOT}/masks" \
  --packed_root "${VIDEOGEN_ROOT}/packed"











# 新终端单独重跑 Step4 前，先执行这段（宏变量+环境）

# 千万千万不要运行 bash docker/run_container.sh
docker start cari4d
docker exec -it cari4d bash


python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip show smplfitter || python -m pip install --no-build-isolation smplfitter
python -c "from smplfitter.pt import BodyModel, BodyFitter; print('smplfitter ok')"

cd /home/warner/_projects/CARI4D
set -o pipefail
report_error(){ c="${1:-ERR}"; shift || true; echo "[ERROR][$c] ${*:-unknown error}"; }
run_or_report(){ "$@"; c=$?; [[ $c -eq 0 ]] || report_error "$c" "cmd failed: $*"; return 0; }

SEQ_RAW="bike_on_may_29_21_17"
SEQ="Date03_Sub01_bike_on_may_29_21_17"
VIDEOGEN_ROOT="data/cari4d-demo/videogen"
VIDEO_OUT="${VIDEOGEN_ROOT}/videos/${SEQ}.0.color.mp4"

echo "VIDEO_OUT=$VIDEO_OUT"
run_or_report test -f "$VIDEO_OUT"




# Minimal texture fallback before demo-custom
run_or_report env SEQ_ENV="$SEQ" python - <<'PY'
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
        ids=[int((t.split('/')[0] if '/' in t else t)) for t in s.split()[1:] if (t.split('/')[0] if '/' in t else t)]
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

export PYTHONPATH=/tmp:${PYTHONPATH:-}
bash scripts/demo-custom.sh "${VIDEO_OUT}"
