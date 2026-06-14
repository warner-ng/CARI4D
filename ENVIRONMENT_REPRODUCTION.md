# CARI4D Environment Reproduction

This file records the local environments used by this checkout on 2026-06-14.
It is intended for moving the project to another machine without guessing package versions.

## Exported files

The full exported environment files are under `env_exports/`:

- `env_exports/cari4d-conda-env.yml`: full conda export for the main CARI4D/Sapiens environment.
- `env_exports/cari4d-pip-freeze.txt`: pip freeze for `cari4d`.
- `env_exports/hy3d-conda-env.yml`: full conda export for Hunyuan3D mesh generation.
- `env_exports/hy3d-pip-freeze.txt`: pip freeze for `hy3d`.
- `env_exports/sam3-conda-env.yml`: full conda export for the SAM3 mask stage.
- `env_exports/sam3-pip-freeze.txt`: pip freeze for `sam3`.
- `env_exports/docker-cari4d-info.json`: Docker engine info plus image/container inspect for `xiexh20/cari4d:latest` and container `cari4d`.

## Runtime environments used by this project

`run_carry_bike_one_shot.sh` and `run_carry_bike_one_shot_full.sh` use these local environments:

| Stage | Environment | Current key versions |
| --- | --- | --- |
| Mask generation | `sam3` | Python 3.12.13, torch 2.5.1+cu121, CUDA runtime 12.1, numpy 1.26.4, OpenCV 4.13.0, h5py 3.16.0 |
| Hunyuan3D mesh | `hy3d` | Python 3.10.20, torch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA runtime 12.1, numpy 2.2.6, OpenCV 4.13.0, h5py 3.16.0 |
| CARI4D/Sapiens | `cari4d` | Python 3.10.20, torch 2.6.0+cu124, torchvision 0.21.0+cu124, torchaudio 2.6.0+cu124, CUDA runtime 12.4, numpy 2.2.6, OpenCV 4.12.0, h5py 3.16.0 |
| CARI4D Docker pipeline | `xiexh20/cari4d:latest` | Image id `sha256:dd0e22363e49939fce428672fe10489c2402a00295736e6a5a2ad10939ca7a9a`, CUDA 12.4.1, cuDNN 9.1.0.70, NCCL 2.21.5 |

Docker on this machine:

- Docker Engine: `Docker version 29.1.3, build 29.1.3-0ubuntu3~22.04.2`
- Docker Compose plugin: not installed
- `docker-compose`: not installed
- Local image tags: `xiexh20/cari4d:latest`, `cari4d:latest`
- Local digest: `xiexh20/cari4d@sha256:dd0e22363e49939fce428672fe10489c2402a00295736e6a5a2ad10939ca7a9a`

## Recreate on another machine

Use the same repo path if possible:

```bash
git clone <repo-url> /home/warner/_projects/CARI4D
cd /home/warner/_projects/CARI4D
```

Recreate conda environments from the exported files:

```bash
conda env create -f env_exports/sam3-conda-env.yml
conda env create -f env_exports/hy3d-conda-env.yml
conda env create -f env_exports/cari4d-conda-env.yml
```

If the target machine already has an environment with the same name, remove or rename it first. The exported yml files include this machine's `prefix`; if conda refuses to create the environment on a different path, delete only the final `prefix:` line from the copied yml file and rerun the same command.

Restore Docker image and container:

```bash
sudo systemctl enable --now docker
docker pull xiexh20/cari4d@sha256:dd0e22363e49939fce428672fe10489c2402a00295736e6a5a2ad10939ca7a9a
docker tag xiexh20/cari4d@sha256:dd0e22363e49939fce428672fe10489c2402a00295736e6a5a2ad10939ca7a9a xiexh20/cari4d:latest
docker tag xiexh20/cari4d:latest cari4d:latest
bash docker/run_container.sh
```

The current container `cari4d` bind-mounts these host paths:

- `/home`
- `/home/warner/_projects/CARI4D`
- `/mnt`
- `/tmp`
- `/tmp/.X11-unix`

The target machine should have equivalent paths or `docker/run_container.sh` must be adjusted before creating the container.

## External files that are not captured by conda or Docker

The README requires these project files separately:

- `unidepth/` cloned from `https://github.com/lpiccinelli-eth/UniDepth.git`
- `VolumetricSMPL/` patched by `scripts/volumetric_smplh.patch`
- `weights/nlf_l_multi_0.3.2.torchscript`
- FoundationPose model weights under `weights/`
- SMPL-H files under `data/smpl/smplh/`
- `data/smpl/kid_template.npy`
- Demo/training data under `data/`
- CoCoNet checkpoint at `experiments/cari4d-release/step031397.pth`

Copy these directories/files from the source machine or download them according to `README.md` before running the pipeline.

## Verification commands

After recreation, run:

```bash
conda run -n sam3 python -c "import torch, cv2, h5py; print(torch.__version__, torch.version.cuda, cv2.__version__, h5py.__version__)"
conda run -n hy3d python -c "import torch, torchvision, cv2, h5py; print(torch.__version__, torchvision.__version__, torch.version.cuda, cv2.__version__, h5py.__version__)"
conda run -n cari4d python -c "import torch, torchvision, torchaudio, cv2, h5py; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda, cv2.__version__, h5py.__version__)"
docker image inspect xiexh20/cari4d:latest --format '{{.Id}} {{json .RepoDigests}}'
docker start cari4d
```
