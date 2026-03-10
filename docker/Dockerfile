FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# System build dependencies needed by Python wheels (for example h5py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    libice6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libvulkan1 \
    mesa-vulkan-drivers \
    pkg-config \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY interactive_world_sim ./interactive_world_sim
COPY main.py ./

# Install build-time Python deps first so source builds succeed reliably.
RUN python -m pip install --upgrade pip wheel "setuptools<81" \
    && pip install "Cython<3" "numpy==1.26.4" pkgconfig \
    && pip install --no-build-isolation -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126/ \
        --extra-index-url https://pypi.nvidia.com \
    && pip install -e .

CMD ["python", "-V"]
