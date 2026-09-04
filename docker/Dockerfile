# =====================================================================
# InstantMesh (TencentARC) — multi-stage build
#
# Stage 1 builds the conda env on a CUDA *devel* base.
# Stage 2 runs it on a *runtime* base + the minimum toolchain that
# nvdiffrast needs, because nvdiffrast compiles JIT on first use.
#
# Build:  docker build -t instantmesh:latest .
# Run:    docker run --rm --gpus all -p 43839:43839 \
#             -v $PWD/models:/workspace/models instantmesh:latest
# =====================================================================


# ---------- Stage 1: build ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata build-essential git wget ca-certificates && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# ---- Miniconda ----
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /workspace/miniconda3 && \
    rm /tmp/miniconda.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -y matters: never rely on stdin not being a TTY
RUN conda create -y -n instantmesh python=3.10 && conda clean -afy

# From here on, "python"/"pip" mean the env's, not base's.
ENV PATH="/workspace/miniconda3/envs/instantmesh/bin:${PATH}"

# ---- CUDA build env (needed for nvdiffrast's --no-build-isolation install) ----
ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/workspace/miniconda3/envs/instantmesh/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"

RUN pip install --no-cache-dir -U pip

# torch first, alone. It pulls the correct triton (2.1.x) as a dependency —
# do NOT install triton explicitly, that silently upgrades it to 3.x and
# breaks xformers.
RUN pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir xformers==0.0.22.post7
RUN pip install --no-cache-dir "numpy<2" "setuptools<81" ninja

WORKDIR /workspace/instantmesh

# Pin everything that a transitive dependency could silently bump.
# numpy and xformers belong here too: rembg/opencv will happily pull numpy 2.x.
RUN printf '%s\n' \
        'torch==2.1.0' \
        'torchvision==0.16.0' \
        'torchaudio==2.1.0' \
        'triton==2.1.0' \
        'xformers==0.0.22.post7' \
        'numpy<2' \
        > /workspace/constraints.txt

COPY ./requirements.txt /workspace/instantmesh/requirements.txt

# nvdiffrast: this only copies .cu/.cpp sources into site-packages.
# The actual compile happens JIT at runtime — see stage 2.
RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/NVlabs/nvdiffrast/ \
        -c /workspace/constraints.txt

# The repo's requirements.txt lists nvdiffrast as its last line; strip it.
# "|| true" guards against grep's exit 1 if that were ever the only line.
RUN (grep -v "nvdiffrast" requirements.txt > requirements_filtered.txt || true) && \
    pip install --no-cache-dir -r requirements_filtered.txt \
        -c /workspace/constraints.txt

# Two gaps in the upstream requirements.txt that break at import time:
#   - rembg no longer pulls onnxruntime (split into extras)
#   - diffusers 0.20.2 imports cached_download, removed in huggingface_hub 0.26
# Plain onnxruntime, not -gpu: the runtime base has no cuDNN, so the GPU
# provider would fail to load and fall back to CPU anyway. u2net is tiny.
RUN pip install --no-cache-dir onnxruntime "huggingface_hub==0.25.2" \
        -c /workspace/constraints.txt

# Fail the build here rather than in production.
RUN python -c "\
import torch, triton, numpy, xformers; \
assert torch.__version__.startswith('2.1.0+cu121'), torch.__version__; \
assert triton.__version__.startswith('2.1'), triton.__version__; \
assert numpy.__version__ < '2', numpy.__version__; \
import nvdiffrast.torch, rembg, diffusers, transformers, gradio, onnxruntime; \
print('stage1 pins + imports OK')"

RUN conda clean -afy && \
    pip cache purge && \
    find /workspace/miniconda3/envs/instantmesh -name "__pycache__" -type d \
        -exec rm -rf {} + 2>/dev/null || true


# ---------- Stage 2: runtime ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
LABEL name="instantmesh" maintainer="instantmesh"

ENV DEBIAN_FRONTEND=noninteractive

# build-essential      -> g++/c++   (torch checks for 'c++' before anything else)
# cuda-nvcc-12-4       -> nvcc
# cuda-cudart-dev-12-4 -> cuda_runtime.h
# The three together are what nvdiffrast's JIT compile needs. None suffices alone.
# The symlink is NOT created automatically when installing these packages
# individually — without it CUDA_HOME points at a directory that doesn't exist.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cuda-nvcc-12-4 \
        cuda-cudart-dev-12-4 \
        libgl1 libglib2.0-0 libegl1 libsm6 libxext6 && \
    ln -sfn /usr/local/cuda-12.4 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    PATH="/workspace/miniconda3/envs/instantmesh/bin:/usr/local/cuda/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX" \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Three caches, three env vars, all under one volume so a --rm run keeps the
# u2net weights and the compiled extension. Note this does NOT cover the
# zero123plus / InstantMesh weights: app.py hardcodes cache_dir='./ckpts/',
# which overrides HF_HOME and puts them in /workspace/instantmesh/ckpts.
ENV HF_HOME=/workspace/models/huggingface \
    U2NET_HOME=/workspace/models/u2net \
    TORCH_EXTENSIONS_DIR=/workspace/models/torch_extensions
VOLUME /workspace/models

COPY --from=builder /workspace/miniconda3/envs/instantmesh /workspace/miniconda3/envs/instantmesh

WORKDIR /workspace/instantmesh

# Pre-baked weight. app.py:97 passes an explicit cache_dir that overrides
# HF_HOME, so this must land in ckpts/ -- which is deliberately NOT under
# the VOLUME above, or the build layer would be discarded.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('TencentARC/InstantMesh', 'diffusion_pytorch_model.bin', repo_type='model', cache_dir='/workspace/instantmesh/ckpts')"

# App code last: editing it doesn't invalidate the env layer above.
COPY . /workspace/instantmesh

# Proves the JIT toolchain actually works, without needing a GPU.
# This is the exact machinery nvdiffrast uses (torch.utils.cpp_extension),
# so if this passes, nvdiffrast will compile.
RUN python -c "\
from torch.utils.cpp_extension import load_inline; \
m = load_inline(name='probe', \
    cpp_sources='int probe_fn() { return 42; }', \
    cuda_sources='__global__ void k(){}', \
    functions=['probe_fn']); \
assert m.probe_fn() == 42; \
print('CUDA toolchain OK')" && \
    rm -rf /workspace/models/torch_extensions/probe

EXPOSE 43839

CMD ["/workspace/miniconda3/envs/instantmesh/bin/python", "app.py"]