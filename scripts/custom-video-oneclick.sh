#!/usr/bin/env bash
set -euo pipefail

# One-click custom video pipeline with compatibility fixes discovered in this session.
# - Converts SMPLH 300-dim npz to CARI4D-compatible pkl (10-dim shapedirs + sparse regressors)
# - Converts open4dhoi outputs to CARI4D masks/packed/mesh inputs
# - Adds minimal texture for object mesh to avoid PyTorch3D texture-batch mismatch
# - Runs robust Docker pipeline to final output/opt/*.pth

usage() {
  cat <<'EOF'
Usage:
  bash scripts/custom-video-oneclick.sh \
    --video /abs/path/to/video.mp4 \
    --seq Date03_Sub01_bike_wild001 \
    --open4dhoi-run /abs/path/to/open4dhoi_code/data/bike/<run_id> \
    --smplh-npz-dir /abs/path/to/smplh_300/smplh

Required:
  --video            Source custom video file
  --seq              Sequence prefix used by CARI4D (e.g. Date03_Sub01_bike_wild001)
  --open4dhoi-run    Folder containing mask_dir/, human_mask_dir/, kp_record_merged.json, obj_org.obj
  --smplh-npz-dir    Folder containing SMPLH_MALE.npz and SMPLH_FEMALE.npz

Optional:
  --fx --fy --cx --cy  Camera intrinsics for wild_video sidecar pkl
                        (defaults: 1410.0195 1365.705 812.03809 548.5144)

Output:
  output/opt/cari4d-release+step031397_demo-hy3d3-optv2/<seq>.pth
EOF
}

VIDEO=""
SEQ=""
OPEN4DHOI_RUN=""
SMPLH_NPZ_DIR=""
FX="1410.0195"
FY="1365.705"
CX="812.03809"
CY="548.5144"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --seq) SEQ="$2"; shift 2;;
    --open4dhoi-run) OPEN4DHOI_RUN="$2"; shift 2;;
    --smplh-npz-dir) SMPLH_NPZ_DIR="$2"; shift 2;;
    --fx) FX="$2"; shift 2;;
    --fy) FY="$2"; shift 2;;
    --cx) CX="$2"; shift 2;;
    --cy) CY="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -f "$VIDEO" ]] || { echo "Missing video: $VIDEO"; exit 1; }
[[ -d "$OPEN4DHOI_RUN" ]] || { echo "Missing open4dhoi run dir: $OPEN4DHOI_RUN"; exit 1; }
[[ -f "$SMPLH_NPZ_DIR/SMPLH_MALE.npz" ]] || { echo "Missing $SMPLH_NPZ_DIR/SMPLH_MALE.npz"; exit 1; }
[[ -f "$SMPLH_NPZ_DIR/SMPLH_FEMALE.npz" ]] || { echo "Missing $SMPLH_NPZ_DIR/SMPLH_FEMALE.npz"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[deps] Check host python dependencies"
if ! python3 - <<'PY' >/dev/null 2>&1
import numpy, scipy, h5py, cv2, joblib
from PIL import Image
PY
then
    echo "[deps] Missing python packages detected, installing..."
    python3 -m pip install --user numpy scipy h5py opencv-python joblib pillow
fi

echo "[1/6] Prepare video paths"
VIDEO_DST="data/cari4d-demo/wild/videos/${SEQ}.0.color.mp4"
mkdir -p "$(dirname "$VIDEO_DST")"
cp -f "$VIDEO" "$VIDEO_DST"

echo "[2/6] Convert SMPLH npz -> compatible pkl"
mkdir -p data/smplh_src data/smpl/smplh data/smpl
cp -f "$SMPLH_NPZ_DIR/SMPLH_MALE.npz" data/smplh_src/SMPLH_MALE.npz
cp -f "$SMPLH_NPZ_DIR/SMPLH_FEMALE.npz" data/smplh_src/SMPLH_FEMALE.npz

python3 - <<'PY'
import numpy as np, pickle, scipy.sparse as sp, shutil, os
os.makedirs('data/smpl/smplh', exist_ok=True)

def norm_scalar(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, np.bytes_):
        return v.tobytes().decode('utf-8')
    if isinstance(v, (bytes, bytearray)):
        return v.decode('utf-8')
    return v

for gender in ['MALE','FEMALE']:
    d = np.load(f'data/smplh_src/SMPLH_{gender}.npz', allow_pickle=True)
    obj = {k: d[k] for k in d.files}

    # CARI4D pipeline expects 10-dim betas
    if isinstance(obj.get('shapedirs'), np.ndarray) and obj['shapedirs'].shape[-1] > 10:
        obj['shapedirs'] = obj['shapedirs'][..., :10].copy()

    # Loader expects sparse regressors with .toarray()
    if isinstance(obj.get('J_regressor'), np.ndarray):
        obj['J_regressor'] = sp.csr_matrix(obj['J_regressor'])
    if isinstance(obj.get('J_regressor_prior'), np.ndarray):
        obj['J_regressor_prior'] = sp.csr_matrix(obj['J_regressor_prior'])

    # Loader expects string, not bytes/0-d arrays
    if 'bs_type' in obj:
        obj['bs_type'] = norm_scalar(obj['bs_type'])
    if 'bs_style' in obj:
        obj['bs_style'] = norm_scalar(obj['bs_style'])

    for name in [f'SMPLH_{gender}.pkl', f'SMPLH_{gender.lower()}.pkl']:
        with open(f'data/smpl/smplh/{name}', 'wb') as f:
            pickle.dump(obj, f, protocol=4)

shutil.copy2('data/smpl/smplh/SMPLH_MALE.pkl', 'data/smpl/SMPLH_male.pkl')
shutil.copy2('data/smpl/smplh/SMPLH_FEMALE.pkl', 'data/smpl/SMPLH_female.pkl')
print('SMPLH compatibility files ready')
PY

echo "[3/6] Build CARI4D masks/packed/mesh from open4dhoi outputs"
SEQ_ENV="$SEQ" OPEN4DHOI_ENV="$OPEN4DHOI_RUN" VIDEO_ENV="$VIDEO_DST" python3 - <<'PY'
import os, json, cv2, h5py, joblib, numpy as np, re
from pathlib import Path
import shutil

seq = os.environ['SEQ_ENV']
src = Path(os.environ['OPEN4DHOI_ENV'])
video = Path(os.environ['VIDEO_ENV'])

masks_root = Path('data/cari4d-demo/videogen/masks')
packed_root = Path('data/cari4d-demo/videogen/packed')
meshes_root = Path('data/cari4d-demo/videogen/meshes')
meshes_metric_root = Path('data/cari4d-demo/videogen/meshes-metric')
for p in [masks_root, packed_root, meshes_root, meshes_metric_root]:
    p.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.release()

obj_dir = src/'mask_dir'
human_dir = src/'human_mask_dir'
src_obj = src/'obj_org.obj'
if (not src_obj.exists()) or src_obj.stat().st_size == 0:
    raise RuntimeError(f'Invalid source obj: {src_obj}. It is missing or empty. Use another --open4dhoi-run with valid obj_org.obj')
obj_files = sorted(obj_dir.glob('*.png'))
human_files = sorted(human_dir.glob('*.png'))
S = min(len(obj_files), len(human_files))
if S == 0:
    raise RuntimeError('No masks found in open4dhoi run')

# masks h5
h5_path = masks_root / f'{seq}_masks_k0.h5'
with h5py.File(h5_path, 'w') as f:
    g = f.create_group(seq)
    obj_stack, hum_stack = [], []
    for i in range(S):
        om = cv2.imread(str(obj_files[i]), cv2.IMREAD_GRAYSCALE)
        hm = cv2.imread(str(human_files[i]), cv2.IMREAD_GRAYSCALE)
        if om.shape != (H, W):
            om = cv2.resize(om, (W, H), interpolation=cv2.INTER_NEAREST)
        if hm.shape != (H, W):
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)
        obj_stack.append(om > 127)
        hum_stack.append(hm > 127)
    for i in range(N):
        j = min(i, S - 1)
        fid = f'{i:06d}'
        g.create_dataset(f'{fid}-k0.obj_rend_mask.png', data=obj_stack[j], compression='gzip')
        g.create_dataset(f'{fid}-k0.person_mask.png', data=hum_stack[j], compression='gzip')

# packed from kp_record_merged.json
raw = json.load(open(src/'kp_record_merged.json'))
frame_map = {k: v for k, v in raw.items() if re.fullmatch(r'\d{5,6}', str(k)) and isinstance(v, dict)}
keys = sorted(frame_map.keys())
if not keys:
    raise RuntimeError('No frame-like records in kp_record_merged.json')
J = 17
joints = np.zeros((N, 1, J, 3), dtype=np.float64)
frames = []
for i in range(N):
    frames.append(f'{i:06d}')
    kk = f'{i:05d}' if f'{i:05d}' in frame_map else (f'{i:06d}' if f'{i:06d}' in frame_map else keys[min(i, len(keys)-1)])
    pts = frame_map[kk].get('2D_keypoint', [])
    valid = []
    if isinstance(pts, list):
        for it in pts:
            if isinstance(it, list) and len(it) >= 2 and isinstance(it[1], (list, tuple)) and len(it[1]) >= 2:
                valid.append((float(it[1][0]), float(it[1][1]), 1.0))
    for j, (x, y, c) in enumerate(valid[:J]):
        joints[i, 0, j] = [x, y, c]
joblib.dump({'frames': frames, 'joints2d': joints}, packed_root / f'{seq}_GT-packed.pkl')

# mesh copy to meshes and meshes-metric
for root in [meshes_root, meshes_metric_root]:
    d = root / f'{seq}_000_rgba'
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f'{seq}_000_align.obj'
    # Important: if dst is a symlink (possibly pointing back to src), unlink first
    # to avoid truncating the source file accidentally.
    if dst.is_symlink():
        dst.unlink()
    shutil.copy2(src_obj, dst)

print('Prepared masks/packed/mesh for', seq)
PY

echo "[4/6] Add minimal texture to metric mesh (PyTorch3D batch consistency)"
SEQ_ENV="$SEQ" python3 - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
import os
seq = os.environ['SEQ_ENV']
obj_path = Path(f'data/cari4d-demo/videogen/meshes-metric/{seq}_000_rgba/{seq}_000_align.obj')
mtl_path = obj_path.with_suffix('.mtl')
tex_path = obj_path.with_name(obj_path.stem + '_tex.png')

verts, faces, raw_v = [], [], []
for line in obj_path.read_text().splitlines():
    if line.startswith('v '):
        raw_v.append(line)
        p = line.split()
        verts.append([float(p[1]), float(p[2]), float(p[3])])
    elif line.startswith('f '):
        idx = [int(t.split('/')[0]) for t in line.split()[1:]]
        if len(idx) >= 3:
            for i in range(1, len(idx)-1):
                faces.append([idx[0], idx[i], idx[i+1]])

if len(verts) == 0 or len(faces) == 0:
    raise RuntimeError(f'Invalid/empty mesh file: {obj_path}. Please check --open4dhoi-run obj_org.obj')

V = np.array(verts)
xy = V[:, :2]
mn, mx = xy.min(0), xy.max(0)
uv = (xy - mn) / (mx - mn + 1e-8)
uv = np.clip(uv, 0, 1)
uv[:, 1] = 1 - uv[:, 1]

with obj_path.open('w') as f:
    f.write(f'mtllib {mtl_path.name}\n')
    f.write('usemtl material_0\n')
    for l in raw_v:
        f.write(l + '\n')
    for u, v in uv:
        f.write(f'vt {u:.6f} {v:.6f}\n')
    for a, b, c in faces:
        f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')

with mtl_path.open('w') as f:
    f.write('newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 2\n')
    f.write(f'map_Kd {tex_path.name}\n')
Image.new('RGB', (4, 4), (200, 200, 200)).save(tex_path)
print('Textured metric mesh ready:', obj_path)
PY

echo "[5/6] Run robust Docker pipeline"
mkdir -p data/cari4d-demo/wild/videos-aligned
cp -f data/cari4d-demo/wild/videos/${SEQ}.0.color.mp4 data/cari4d-demo/wild/videos-aligned/${SEQ}.0.color.mp4

FX_ENV="$FX" FY_ENV="$FY" CX_ENV="$CX" CY_ENV="$CY" PKL_PATH="data/cari4d-demo/wild/videos-aligned/${SEQ}.0.color.pkl" python3 - <<'PY'
import joblib, os
joblib.dump(
    {'fx': float(os.environ['FX_ENV']), 'fy': float(os.environ['FY_ENV']), 'cx': float(os.environ['CX_ENV']), 'cy': float(os.environ['CY_ENV'])},
    os.environ['PKL_PATH']
)
print('Wrote intrinsics sidecar', os.environ['PKL_PATH'])
PY

cat > "$ROOT/.tmp_stage5_inside.sh" <<'EOS'
#!/usr/bin/env bash
set -e

python - <<'PY'
import importlib.util, subprocess, sys
checks = [
    ("smplfitter", "smplfitter"),
    ("chumpy", "chumpy"),
    ("smplx", "smplx"),
    ("transformers", "transformers"),
    ("av", "av"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
]
missing = [pkg for mod, pkg in checks if importlib.util.find_spec(mod) is None]
if missing:
    print("[docker-deps] installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
else:
    print("[docker-deps] all good")
PY

# 5.1 Unidepth + NLF
python prep/unidepth_behave.py --wild_video --video "$VIDEO_DST" -o data/cari4d-demo/wild/videos
python prep/run_nlf_sepK.py -o data/cari4d-demo/videogen/nlf --masks_root data/cari4d-demo/videogen/masks --video "$VIDEO_DST" --wild_video

# 5.2 Fallback nlf-opt (skip fit_smplh_global compatibility pitfalls)
mkdir -p data/cari4d-demo/videogen/nlf-opt
cp -f data/cari4d-demo/videogen/nlf/"$SEQ"_params.pkl data/cari4d-demo/videogen/nlf-opt/"$SEQ"_params.pkl

# 5.3 Prepare aligned folder sidecars used by later steps
cp -f data/cari4d-demo/wild/videos/"$SEQ".0.color.mp4 data/cari4d-demo/wild/videos-aligned/"$SEQ".0.color.mp4
cp -f data/cari4d-demo/wild/videos/"$SEQ".0.depth-reg.mp4 data/cari4d-demo/wild/videos-aligned/"$SEQ".0.depth-reg.mp4

# 5.4 FP tracking
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 --masks_root data/cari4d-demo/videogen/masks --hy3d_root=data/cari4d-demo/videogen/meshes-metric --video data/cari4d-demo/wild/videos-aligned/"$SEQ".0.color.mp4 -o data/cari4d-demo/videogen/fp-hy3d3-track

# 5.5 CoCoNet + optimization
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json use_sel_view=True render_video=True identifier=_demo use_intermediate=False data_name=test-only hy3d_meshes_root=data/cari4d-demo/videogen/meshes-metric masks_root=data/cari4d-demo/videogen/masks fp_root=data/cari4d-demo/videogen/fp-hy3d3-track nlf_root=data/cari4d-demo/videogen/nlf-opt video=data/cari4d-demo/wild/videos-aligned/"$SEQ".0.color.mp4 cam_id=0 wild_video=True outpath=output/coconet
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=16 opt_rot=True opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=0.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False pth_file=output/coconet/cari4d-release+step031397_demo/"$SEQ".pth wild_video=True use_input=True video_root=data/cari4d-demo/wild/videos-aligned packed_root=data/cari4d-demo/videogen/packed masks_root=data/cari4d-demo/videogen/masks hy3d_meshes_root=data/cari4d-demo/videogen/meshes-metric outpath=output/opt
EOS

chmod +x "$ROOT/.tmp_stage5_inside.sh"

sg docker -c "docker run --rm --gpus all --network=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SEQ=$SEQ -e VIDEO_DST=$VIDEO_DST -v $ROOT:/workspace -w /workspace xiexh20/cari4d:latest bash /workspace/.tmp_stage5_inside.sh"

echo "[6/6] Done"
echo "Final output: output/opt/cari4d-release+step031397_demo-hy3d3-optv2/${SEQ}.pth"