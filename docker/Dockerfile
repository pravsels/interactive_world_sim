FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# System build dependencies needed by Python wheels (for example h5py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libgtk-3-0 \
    python3.11 \
    python3.11-dev \
    libhdf5-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    pkg-config \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

COPY pyproject.toml README.md requirements.txt ./
COPY interactive_world_sim ./interactive_world_sim
COPY main.py ./

# Install build-time Python deps first so source builds succeed reliably.
RUN python -m pip install --upgrade pip wheel "setuptools<81" \
    && pip install "Cython<3" "numpy==1.26.4" pkgconfig py-cpuinfo \
    && pip install --no-build-isolation -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126/ \
        --extra-index-url https://pypi.nvidia.com \
    && pip install -e .

CMD ["python", "-V"]
