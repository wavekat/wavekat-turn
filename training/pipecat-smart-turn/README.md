# Training Pipecat Smart Turn

Upstream repo: <https://github.com/pipecat-ai/smart-turn>

## Model Overview

- **Architecture:** Whisper Tiny encoder + attention-pooling classification head (~8M params)
- **Task:** Binary classification — complete vs. incomplete turn
- **Input:** 16 kHz mono PCM, up to 8 seconds, log-mel spectrogram (80×800)
- **Loss:** BCEWithLogitsLoss with dynamic positive-weight balancing
- **Output:** ONNX (FP32 ~32 MB, INT8 ~8 MB)

## Infrastructure

- **Region:** Azure Australia East
- **VM Type:** NC4as_T4_v3 (4 vCPUs, 28 GB RAM, NVIDIA Tesla T4 16 GB)
- **Hostname:** `gpu-testing` (via Tailscale)
- **User:** `eason`
- **SSH:** `ssh gpu-testing` (key: `~/.ssh/id_ed25519_wavekat-eason`, configured in `~/.ssh/config`)
- **GPU:** Tesla T4, 16 GB VRAM, driver 590.48.01, CUDA 13.1

## Steps

### 1. Connect to Azure VM

```bash
ssh gpu-testing
```

### 2. Environment Setup

#### 2.1 GPU Driver

```bash
sudo apt update
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y nvidia-driver-590
sudo reboot

# If secure boot blocks the module:
sudo mokutil --disable-validation
sudo reboot

# Verify
nvidia-smi
```

#### 2.2 Disk Setup

Two additional data disks mounted for datasets and checkpoints:

```bash
sudo mkfs.ext4 /dev/sdc
sudo mkfs.ext4 /dev/sdd
sudo mkdir -p /datasets /checkpoints
sudo mount /dev/sdc /datasets
sudo mount /dev/sdd /checkpoints

# Persist in fstab
BLK_SDC=$(sudo blkid -s UUID -o value /dev/sdc)
BLK_SDD=$(sudo blkid -s UUID -o value /dev/sdd)
echo "UUID=$BLK_SDC /datasets ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
echo "UUID=$BLK_SDD /checkpoints ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

#### 2.3 Docker + NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure runtime and move Docker root to data disk
sudo systemctl stop docker
sudo mkdir -p /datasets/docker
sudo rsync -aP /var/lib/docker/ /datasets/docker/
sudo nvidia-ctk runtime configure --runtime=docker
# Edit /etc/docker/daemon.json to add: "data-root": "/datasets/docker"
sudo systemctl start docker
sudo rm -rf /var/lib/docker

# Move containerd storage to data disk as well
# (Docker's data-root only moves Docker's own data, not containerd's.
#  Without this, containerd fills up the root disk during image builds.)
sudo systemctl stop docker
sudo systemctl stop containerd
sudo mkdir -p /datasets/containerd
sudo rsync -aP /var/lib/containerd/ /datasets/containerd/
sudo rm -rf /var/lib/containerd
sudo ln -s /datasets/containerd /var/lib/containerd
sudo systemctl start containerd
sudo systemctl start docker

# Verify GPU in container
docker run --rm --gpus all nvidia/cuda:13.1.0-devel-ubuntu24.04 nvidia-smi
```

### 3. Fix Disk Permissions

```bash
sudo chown $USER:$USER /checkpoints /datasets
```

### 4. Build Docker Image

**Local machine** — copy the Dockerfile to the VM:

```bash
scp training/pipecat-smart-turn/Dockerfile gpu-testing:/checkpoints/Dockerfile.smart-turn
```

**On the VM** — build the image:

```bash
docker build -t smart-turn -f /checkpoints/Dockerfile.smart-turn /checkpoints
```

The Dockerfile lives in this repo at `training/pipecat-smart-turn/Dockerfile`.
`/checkpoints` and `/datasets` are kept clean as pure data volumes.

### 5. Data Preparation

Upstream datasets (HuggingFace):

- **Train:** `pipecat-ai/smart-turn-data-v3.2-train` (270k samples, ~41 GB)
- **Test:** `pipecat-ai/smart-turn-data-v3.2-test`

Dataset columns: `audio`, `id`, `language`, `endpoint_bool`, `midfiller`, `endfiller`, `synthetic`, `dataset`

Pre-download the dataset so it persists across container runs:

```bash
docker run -d \
  --name smart-turn-download \
  --gpus all \
  -v /datasets/huggingface:/root/.cache/huggingface \
  smart-turn \
  python -c "from datasets import load_dataset; load_dataset('pipecat-ai/smart-turn-data-v3.2-train')"
```

The data is cached at `/datasets/huggingface` on the host. All subsequent runs
must mount this path to `/root/.cache/huggingface` to reuse it.

Raw data format for custom contributions:
- FLAC files, mono, 16-bit, 16 kHz+
- Max 16 seconds per file, ~200 ms trailing silence
- Directory structure: `{language}/{complete|incomplete}-{midfiller|endfiller|nofiller}/{uuid}.flac`
- Convert raw to HF dataset: `python datasets/scripts/raw_to_hf_dataset.py <base_name> <input_dir> <output_dir> <tmp_dir>`

### 6. Dataset Exploration (Notebook)

**Local machine** — copy the notebook to the VM:

```bash
scp training/pipecat-smart-turn/explore_dataset.ipynb gpu-testing:/checkpoints/explore_dataset.ipynb
```

**On the VM** — launch JupyterLab:

```bash
docker run -d --name jupyter \
  --gpus all --restart unless-stopped \
  -v /datasets/huggingface:/root/.cache/huggingface \
  -v /checkpoints:/checkpoints \
  -p 8888:8888 \
  smart-turn \
  jupyter lab \
    --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --notebook-dir=/checkpoints \
    --ServerApp.token='' --ServerApp.password=''
```

Open `http://gpu-testing:8888` in a browser (via Tailscale) and run
`explore_dataset.ipynb`. The notebook covers label balance, audio durations,
language/filler/synthetic breakdowns, audio playback, and mel spectrogram
visualisation.

### 7. Training

```bash
docker run --gpus all \
  -v /datasets/huggingface:/root/.cache/huggingface \
  -v /checkpoints:/checkpoints \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  smart-turn \
  python train_local.py \
    --training-run-name my-run \
    --output-dir /checkpoints/output
```

Hyperparameters (defaults in `train.py`):

| Param | Value |
|---|---|
| Base model | `openai/whisper-tiny` |
| Learning rate | 5e-5 |
| Epochs | 4 |
| Train batch size | 384 |
| Eval batch size | 128 |
| Warmup ratio | 0.2 |
| Weight decay | 0.01 |
| LR schedule | Cosine |
| Eval/save steps | 500 |
| Dataloader workers | 6 |

Optional: set `WANDB_API_KEY` for experiment tracking.

> **Note:** Batch size 384 requires significant VRAM. With T4 (16 GB) we will
> likely need to reduce this — experiment with 32–64.

### 8. Quantization

```bash
docker run --gpus all \
  -v /checkpoints:/checkpoints \
  smart-turn \
  python train_local.py --quantize /checkpoints/output/path-to-fp32-model.onnx
```

INT8 static quantization using entropy calibration on 1024 samples.

### 9. Benchmarking

```bash
docker run --gpus all \
  -v /checkpoints:/checkpoints \
  smart-turn \
  python train_local.py --benchmark /checkpoints/output/
```

Reference latencies: CPU ~12.6 ms, GPU (L40S) ~3.3 ms.

### 10. Export / Integration

Final artifacts:
- `smart-turn-v3.onnx` (FP32)
- `smart-turn-v3-int8.onnx` (INT8)

TODO: Integrate into wavekat-turn.
