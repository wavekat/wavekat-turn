FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    portaudio19-dev git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace

# Clone upstream repo
RUN git clone https://github.com/pipecat-ai/smart-turn.git .

# Install Python dependencies
RUN python -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Default: show usage
CMD ["python", "train_local.py", "--help"]
