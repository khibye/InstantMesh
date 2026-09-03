# ---------- Stage 1: build ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata build-essential git wget unzip && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n instantmesh python=3.10
ENV PATH="/workspace/miniconda3/envs/instantmesh/bin:${PATH}"

RUN conda install Ninja
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.22.post7 triton
RUN pip install "numpy<2" "setuptools<81" ninja
RUN pip install onnxruntime-gpu

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

WORKDIR /workspace/instantmesh
ADD ./requirements.txt /workspace/instantmesh/requirements.txt

# lock torch so nothing below silently upgrades it
RUN echo "torch==2.1.0" > constraints.txt && \
    echo "torchvision==0.16.0" >> constraints.txt && \
    echo "torchaudio==2.1.0" >> constraints.txt

RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/ -c constraints.txt
RUN grep -v "nvdiffrast" requirements.txt > requirements_filtered.txt && \
    pip install -r requirements_filtered.txt -c constraints.txt
RUN pip install "huggingface_hub==0.25.2" -c constraints.txt

# sanity check before freezing this layer — must print 2.1.0+cu121
RUN python -c "import torch; print('torch:', torch.__version__)"

# clean caches before copying forward
RUN conda clean -afy && \
    pip cache purge && \
    find /workspace/miniconda3/envs/instantmesh -name "*.a" -delete && \
    find /workspace/miniconda3/envs/instantmesh -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ---------- Stage 2: runtime ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
LABEL name="instantmesh" maintainer="instantmesh"
VOLUME /workspace/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends libegl1-mesa-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# bring over the fully-built conda env — already has torch, xformers, triton, nvdiffrast, everything
COPY --from=builder /workspace/miniconda3/envs/instantmesh /workspace/miniconda3/envs/instantmesh
ENV PATH="/workspace/miniconda3/envs/instantmesh/bin:${PATH}"

WORKDIR /workspace/instantmesh

# app code copied last so code changes don't invalidate the (already-clean) env above
COPY . /workspace/instantmesh

CMD ["python", "app.py"]