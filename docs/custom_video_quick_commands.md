# Custom video quick commands (from video to `output/opt/*.pth`)

This is the shortest command flow we validated in this session.

For full troubleshooting history and all compatibility fixes used in the successful run, see:
`docs/custom_video_full_runbook_20260529.md`.

## 1) Make script executable

```bash
chmod +x scripts/custom-video-oneclick.sh
```

## 2) One-click run

```bash
bash scripts/custom-video-oneclick.sh \
  --video /home/warner/_projects/CARI4D/carrying_bike.mov \
  --seq Date03_Sub01_bike_wild001 \
  --open4dhoi-run /home/warner/_projects/open4dhoi_code/data/bike/20260522_233018_eaf586b7 \
  --smplh-npz-dir /home/warner/_projects/smplh_try/3_smplh_300/smplh
```

## 3) Final outputs

- CoCoNet result:
  - `output/coconet/cari4d-release+step031397_demo/Date03_Sub01_bike_wild001.pth`
- Final optimized result:
  - `output/opt/cari4d-release+step031397_demo-hy3d3-optv2/Date03_Sub01_bike_wild001.pth`
- Optimization visualizations:
  - `output/opt/cari4d-release+step031397_demo-hy3d3-optv2/Date03_Sub01_bike_wild001+step*.mp4`

## Notes

- This command path intentionally includes compatibility workarounds discovered during debugging:
  - SMPLH 300-dim `.npz` conversion and compatibility normalization
  - open4dhoi mask/keypoint/mesh conversion for CARI4D custom format
  - texture consistency fix for object mesh used by PyTorch3D batching
  - lower-memory optimization settings for 16GB-class GPUs
- If your camera intrinsics differ, pass `--fx --fy --cx --cy` to the one-click script.
