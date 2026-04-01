# Training Pipecat Smart Turn

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
# Edit /etc/docker/daemon.json to set "data-root": "/datasets/docker"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl start docker
sudo rm -rf /var/lib/docker

# Verify GPU in container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 3. Data Preparation

TODO: Dataset sourcing, preprocessing, format.

### 4. Training

TODO: Training script, hyperparameters, config.

### 5. Evaluation

TODO: Metrics, validation steps.

### 6. Export / Integration

TODO: Convert model for use in wavekat-turn.
