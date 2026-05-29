# Processing Custom Video

## One-shot commands (Step1 → Step4)

```bash
set -o pipefail
report_error(){ c="${1:-ERR}"; shift || true; echo "[ERROR][$c] ${*:-unknown error}"; }
run_or_report(){ "$@"; c=$?; [[ $c -eq 0 ]] || report_error "$c" "cmd failed: $*"; return 0; }
cd /home/warner/_projects/CARI4D

HUMAN_PROMPT="man"
OBJECT_PROMPT="bike"


SEQ_RAW="trim_carry_bike"
SEQ="trim_Sub01_bike_wild001"
VIDEO_IN="data/trim_carry_bike/videos/trim_carry_bike.0.color.mp4"
MASKS_IN="data/trim_carry_bike/masks/trim_carry_bike_masks_k0.h5"
BLENDER_PATH="/home/warner/tools/blender-3.6.17-linux-x64/blender"

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
mkdir -p "${VIDEOGEN_ROOT}/videos" "${VIDEOGEN_ROOT}/meshes-metric" "${VIDEOGEN_ROOT}/fp-hy3d3-track" "${VIDEOGEN_ROOT}/packed"

source /home/warner/miniconda3/etc/profile.d/conda.sh
conda activate sam3
HF_TOKEN="$(hf auth token)"
export HF_TOKEN
run_or_report python prep/run_sam3_masks.py \
  --video "$VIDEO_IN" \
  --human_prompt "$HUMAN_PROMPT" \
  --object_prompt "$OBJECT_PROMPT" \
  --visualize












run_or_report test -f "$MASKS_IN"

src_v="$(realpath "$VIDEO_IN")"
ln -sfn "$src_v" "$VIDEO_OUT"

src_m="$(realpath "$MASKS_IN")"
ln -sfn "$src_m" "$MASKS_OUT"

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






cd /home/warner/_projects/CARI4D
source /home/warner/miniconda3/etc/profile.d/conda.sh
conda activate cari4d
unset LD_LIBRARY_PATH

# 只在 group 名不一致时做一次
python - <<'PY'
import h5py
p="data/cari4d-demo/videogen/masks/trim_Sub01_bike_wild001_masks_k0.h5"
old="trim_carry_bike"
new="trim_Sub01_bike_wild001"
with h5py.File(p,"r+") as f:
    if old in f and new not in f:
        f.copy(old, new); del f[old]
print("ok")
PY

run_or_report conda run -n cari4d python prep/run_sapiens_pose.py \
  --video "$VIDEO_OUT" \
  --masks_root "${VIDEOGEN_ROOT}/masks" \
  --packed_root "${VIDEOGEN_ROOT}/packed"










# 新终端单独重跑 Step4 前，先执行这段（宏变量+环境）

# 千万千万不要运行 bash docker/run_container.sh

docker exec -it cari4d bash


python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip show smplfitter || python -m pip install --no-build-isolation smplfitter
python -c "from smplfitter.pt import BodyModel, BodyFitter; print('smplfitter ok')"


cd /home/warner/_projects/CARI4D
set -o pipefail
report_error(){ c="${1:-ERR}"; shift || true; echo "[ERROR][$c] ${*:-unknown error}"; }
run_or_report(){ "$@"; c=$?; [[ $c -eq 0 ]] || report_error "$c" "cmd failed: $*"; return 0; }



SEQ_RAW="trim_carry_bike"
SEQ="trim_Sub01_bike_wild001"
VIDEOGEN_ROOT="data/cari4d-demo/videogen"
VIDEO_OUT="${VIDEOGEN_ROOT}/videos/${SEQ}.0.color.mp4"

echo "VIDEO_OUT=$VIDEO_OUT"
run_or_report test -f "$VIDEO_OUT"





bash scripts/demo-custom.sh data/cari4d-demo/videogen/videos/trim_Sub01_bike_wild001.0.color.mp4










Pre-processed examples are available from [this file](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip). Download and unzip:

```bash
unzip generated-videos.zip -d data/videogen
bash scripts/demo-custom.sh data/cari4d-demo/videogen/videos/Date03_Sub01_Suitcase_Dragging-wild.0.color.mp4
```

For your own RGB videos, see `data/cari4d-demo/` for the expected data layout.

> **Note:** Our method is not designed for partially visible bodies or long-term occlusions. It works best when both the person and object are mostly visible. See the teaser videos on our [website](https://nvlabs.github.io/CARI4D/) and [generated videos](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip) for examples.

---

## Step 1: Human and Object Masks

Prepare masks as a packed HDF5 file. Each frame needs a human mask and an object mask.

### Output format

`<masks_root>/<seq>_masks_k<kid>.h5` — HDF5 file with a top-level group named after the sequence (e.g. `Date03_Sub01_gas_wild002`). Each frame contributes two datasets:

| Dataset key | Type | Shape | Description |
|---|---|---|---|
| `<frame_id>-k<kid>.person_mask.png` | `bool` | `(H, W)` | Human binary mask |
| `<frame_id>-k<kid>.obj_rend_mask.png` | `bool` | `(H, W)` | Object binary mask |

- `<frame_id>` is a 6-digit zero-padded frame index (`000000`, `000001`, ...)
- For in-the-wild videos, use `kid=0`
- Masks are loaded by [this function](https://github.com/NVlabs/CARI4D/blob/main/behave_data/behave_video.py#L23-L45)

### Using SAM3 (recommended)

We provide `prep/run_sam3_masks.py` which uses [SAM3](https://github.com/facebookresearch/sam3) for text-prompted video segmentation. It takes a video and two text prompts (human + object), segments and tracks both across all frames, and saves the result in the HDF5 format above.

**Setup** (requires Python 3.12+, PyTorch 2.7+, CUDA — use a separate env from CARI4D):

```bash
# 1. Clone SAM3 into project root
git clone https://github.com/facebookresearch/sam3.git

# 2. Install SAM3 and dependencies (in a Python 3.12+ env)
cd sam3 && pip install -e . && pip install einops h5py opencv-python pycocotools psutil imageio
cd ..

# 3. Authenticate with HuggingFace for checkpoint access
#    Request access at https://huggingface.co/facebook/sam3, then:
huggingface-cli login --token $HF_TOKEN
#    Checkpoints (sam3.pt from facebook/sam3) are auto-downloaded on first run.
```

**Example:**

```bash
python prep/run_sam3_masks.py \
    --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
  --human_prompt "<human_prompt>" \
  --object_prompt "<object_prompt>" \
    --visualize
```

**Notes:**
- Processes video in chunks (default 300 frames, configurable via `--chunk_size`) to fit within 24GB GPU memory.
- Use short text prompts (e.g. `"man"`, `"person"`) for the human — long descriptive prompts may fail detection in some chunks.
- Add `--visualize` to save a side-by-side MP4 (RGB | RGB + mask overlay) for inspection.

---

## Step 2: Object Reconstruction

Use Hunyuan3D or SAM3D to reconstruct the object mesh. Place the output under `<hy3d_root>`. Step by step instruction:

- Extract first frame RGB and object mask, remove the background, and crop a square around the object with 0.20 border margin, save the result image as one RGBA png file. Note if object is heavily occluded in the first frame, we recommend using another frame in the image for better object reconstruction. 
- Run Hunyuan3D/SAM3D with the RGBA image as input,  reconstruct 3D and convert glb to obj file. 
- **Mesh path convention:** `<seq>*_<frame_index:03d>_rgba/<seq>*_<frame_index:03d>_align.obj`
  where `frame_index` is the video frame used for reconstruction.
- The mesh should be in normalized scale (longest axis in `[-1, 1]`,  direct Hunyuan3D output result). Metric-scale estimation is done later using UniDepth.

### Example with `run_hy3d_recon.py` 

We provide `prep/run_hy3d_recon.py` which automates the full pipeline: RGBA extraction from video + masks, Hunyuan3D shape and texture generation, and GLB-to-OBJ conversion.

**Setup** (use a separate env from CARI4D, requires diffusers 0.31.0):

```bash
# 1. Create and activate a Hunyuan3D environment
conda create -n hy3d python=3.10 && conda activate hy3d

# 2. Install Hunyuan3D-2 (follow https://github.com/Tencent/Hunyuan3D-2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/Tencent/Hunyuan3D-2.git
# Or clone and install locally:
# git clone https://github.com/Tencent/Hunyuan3D-2.git && cd Hunyuan3D-2 && pip install -e .

# 3. Install additional dependencies
pip install h5py opencv-python

# 4. Ensure Blender 3.6+ is available for GLB-to-OBJ conversion
#    Download from https://www.blender.org/download/ if needed
```

**Example:**

```bash
python prep/run_hy3d_recon.py \
    --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
    --masks_root data/cari4d-demo/wild/masks \
    --hy3d_root data/cari4d-demo/meshes \
    --blender_path /path/to/blender

python prep/run_hy3d_recon.py \
  --video data/carry_bike/videos/carry_bike.0.color.mp4 \
  --masks_root data/carry_bike/masks \
  --hy3d_root data/carry_bike/meshes \
  --blender_path blender
```

**Notes:**
- By default uses frame 0 for reconstruction. If the object is heavily occluded in the first frame, specify `--frame_index N` to use a different frame.
- Use `--skip_hy3d` to only extract the RGBA image (e.g. for manual inspection before running reconstruction).
- Use `--skip_glb2obj` to skip the Blender conversion step (e.g. if you want to inspect the GLB first).
- The script skips processing if the output OBJ already exists.

---

## Step 3: 2D Human Keypoints

Run 2D human body keypoint detection and pack the results into a pkl file.

### Output format

`<packed_root>/<seq>_GT-packed.pkl` (saved with `joblib.dump`) — a dict with the following keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `frames` | `list[str]` | `[N]` | 6-digit zero-padded frame indices, e.g. `['000000', '000001', ...]` |
| `joints2d` | `ndarray` | `(N, K, J, 3)` float64 | 2D keypoints per frame. `K` = number of views (1 for in-the-wild); `J` = 17 (COCO) or 25 (OpenPose); last dim = `(x, y, confidence)` |

The pipeline auto-detects the keypoint format from the `J` dimension — no configuration flag is needed.

### Option A: Sapiens (recommended)

[Sapiens](https://github.com/facebookresearch/sapiens) predicts COCO 17 keypoints and runs inside the `cari4d` conda env.

**Important**: You need to download the [joint regressor](https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/data/COCO/J_regressor_coco.npy) to use COCO 17 keypoints, place it under folder `data/assets`. 

**Setup:**

```bash
# 1. Clone Sapiens into the project root
git clone https://github.com/facebookresearch/sapiens.git

# 2. Install dependencies (in the cari4d conda env)
conda activate cari4d
pip install mmcv-lite mmengine mmdet mmpretrain xtcocotools json_tricks munkres

# 3. Authenticate with HuggingFace for checkpoint access
#    Request access at https://huggingface.co/noahcao/sapiens-pose-coco, then:
huggingface-cli login --token $HF_TOKEN

# 4. Download the Sapiens 0.3b pose checkpoint
mkdir -p ~/sapiens_host/pose/checkpoints/sapiens_0.3b
huggingface-cli download noahcao/sapiens-pose-coco \
    sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth \
    --local-dir ~/sapiens_host/pose/checkpoints
#    Alternatively, use the 0.6b model for higher accuracy (requires more VRAM):
#    huggingface-cli download noahcao/sapiens-pose-coco \
#        sapiens_0.6b/sapiens_0.6b_coco_best_coco_AP_812.pth \
#        --local-dir ~/sapiens_host/pose/checkpoints
```

**Example:**

```bash
python prep/run_sapiens_pose.py \
    --video data/cari4d-demo/wild/videos/<seq>.0.color.mp4 \
    --masks_root data/cari4d-demo/wild/masks \
    --packed_root data/cari4d-demo/wild/packed-coco
```

The script uses person masks to compute bounding boxes for top-down pose estimation. See `prep/run_sapiens_pose.py` for additional options (`--checkpoint`, `--batch_size`, `--device`).

### Option B: OpenPose

Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect Body 25 keypoints, and pack the results into the same pkl format above. See the OpenPose documentation for installation and usage.

---

## Step 4: Run the Pipeline

Run the full CARI4D pipeline. The video file should be an MP4 (`<seq>.0.color.mp4`) placed under `data/cari4d-demo/wild/videos/`.

```bash
bash scripts/demo-custom.sh data/cari4d-demo/wild/videos/<seq>.0.color.mp4
```

Update `packed_root` in the script to point to your keypoint output directory (e.g. `packed-coco` for Sapiens).
