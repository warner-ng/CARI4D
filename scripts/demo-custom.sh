# Run CARI4D on custom videos.
# Please follow ./docs/custom_videos.md to prepare data before running this script.
# Required data before running: 1). Object mesh. 2). Masks of human and object. 3). openpose detections of the human. 

video=$1
video_prefix=$(basename "$video" | cut -d. -f1)
video_dir=$(dirname "$video")
echo $video_prefix

# Use experiment-start timestamp to avoid overwriting artifacts across runs.
# You can override manually: EXP_TS=20260529-153000 bash scripts/demo-custom.sh <video>
EXP_TS=${EXP_TS:-$(date +"%Y%m%d-%H%M%S")}
RUN_IDENTIFIER="_demo_${EXP_TS}"
OPT_SAVE_NAME="optv2_${EXP_TS}"
COCONET_SAVE_DIR="cari4d-release+step031397${RUN_IDENTIFIER}"
echo "EXP_TS=${EXP_TS}"

# Final pipeline profile:
#   - wild_robust (default): aligned with wild/oneclick scripts for custom in-the-wild videos
#   - behave_like_demo: aligned with scripts/demo.sh settings
# You can still override any value via env vars below.
PIPELINE_PROFILE=${PIPELINE_PROFILE:-wild_robust}
if [[ "${PIPELINE_PROFILE}" == "behave_like_demo" ]]; then
    COCONET_USE_INTERMEDIATE_DEFAULT=True
    OPT_REFINE_BS_DEFAULT=32
    OPT_REFINE_PEN_DEFAULT=2.0
    OPT_W_J2D_DEFAULT=0.03
else
    COCONET_USE_INTERMEDIATE_DEFAULT=False
    OPT_REFINE_BS_DEFAULT=16
    OPT_REFINE_PEN_DEFAULT=0.0
    OPT_W_J2D_DEFAULT=0.006
fi

COCONET_USE_INTERMEDIATE=${COCONET_USE_INTERMEDIATE:-${COCONET_USE_INTERMEDIATE_DEFAULT}}
OPT_REFINE_BS=${OPT_REFINE_BS:-${OPT_REFINE_BS_DEFAULT}}
OPT_REFINE_PEN=${OPT_REFINE_PEN:-${OPT_REFINE_PEN_DEFAULT}}
OPT_W_J2D=${OPT_W_J2D:-${OPT_W_J2D_DEFAULT}}
COCONET_CLIP_LEN=${COCONET_CLIP_LEN:-8}

# Paths that store preprocessed data:
masks_root=data/cari4d-demo/videogen/masks/ # store the masks of human and object.
packed_root=data/cari4d-demo/videogen/packed/ # store the openpose detections for each frame.
hy3d_root=data/cari4d-demo/videogen/meshes # store the reconstructed object mesh in normalized scale. 

# Paths for intermediate results: 
nlf_path=data/cari4d-demo/videogen/nlf
fp_root=data/cari4d-demo/videogen/fp-hy3d3-track
coconet_out=output/coconet

set -e

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --wild_video --video ${video} -o ${video_dir}

# Step 2: run NLF
python prep/run_nlf_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video

# Step 2 (alternative): run SAM3D-body instead of NLF
# python prep/run_sam3d_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video

# Step 3: run SMPLH fitting to get globally consistent human pose and translation
python prep/fit_smplh_global.py --wild_video --video ${video} --packed_root ${packed_root} --masks_root ${masks_root} \
    --nlf_path=${nlf_path} -o ${nlf_path}-opt

# Step 4: align Unidepth to GENMO human
python prep/align_monod2hum.py --wild_video --nlf_path ${nlf_path}-opt \
--masks_root ${masks_root} \
--video ${video}

# Update the video path, pointing to the new video with aligned depth. 
video=${video_dir}-aligned/${video_prefix}.0.color.mp4 

# Step 5: estimate metric scale of the object 
python tools/estimate_scale_video.py --wild_video --video ${video} --masks_root ${masks_root} --hy3d_root ${hy3d_root} -o ${hy3d_root}-metric


# Step 5: run FP in tracking mode
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root ${masks_root} --hy3d_root=${hy3d_root}-metric \
--video ${video} -o ${fp_root}

# Apply memory allocator settings before CoCoNet and optimization.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# Step 6: run CoCoNet to refine human + object
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=${RUN_IDENTIFIER} use_intermediate=${COCONET_USE_INTERMEDIATE} data_name=test-only \
hy3d_meshes_root=${hy3d_root}-metric \
masks_root=${masks_root} \
fp_root=${fp_root} \
nlf_root=${nlf_path}-opt \
video=${video}  cam_id=0 wild_video=True \
outpath=${coconet_out} \
clip_len=${COCONET_CLIP_LEN}

# Step 7: run joint optimization
# Override when needed, e.g. OPT_REFINE_BS=96 OPT_REFINE_PEN=2.0 OPT_W_J2D=0.03 bash scripts/demo-custom.sh <video>
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=${OPT_SAVE_NAME} batch_size=${OPT_REFINE_BS} opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=${OPT_REFINE_PEN} w_j2d=${OPT_W_J2D} opt_smpl_trans=False opt_betas=False  \
pth_file=${coconet_out}/${COCONET_SAVE_DIR}/${video_prefix}.pth  wild_video=True use_input=True \
video_root=$(dirname "$video") \
packed_root=${packed_root} \
masks_root=${masks_root}  \
hy3d_meshes_root=${hy3d_root}-metric outpath=output/opt
# Note: if OOM persists, reduce OPT_REFINE_BS further.

echo "PIPELINE_PROFILE=${PIPELINE_PROFILE}"
echo "CoCoNet use_intermediate=${COCONET_USE_INTERMEDIATE}"
echo "CoCoNet dir: ${coconet_out}/${COCONET_SAVE_DIR}"
echo "Opt save_name: ${OPT_SAVE_NAME}"
echo "Opt params: batch_size=${OPT_REFINE_BS}, w_pen=${OPT_REFINE_PEN}, w_j2d=${OPT_W_J2D}"