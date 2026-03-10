#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE_NAME="${IMAGE_NAME:-interactive-world-sim}"
IMAGE_TAG="${IMAGE_TAG:-local}"
DOCKER_ARGS=(
  --rm
  -it
  --gpus all
  -v "${REPO_DIR}:/workspace"
  -w /workspace
)

# Forward X11 display access when available for GUI/OpenCV workflows.
if [ -n "${DISPLAY:-}" ] && [ -d "/tmp/.X11-unix" ]; then
  DOCKER_ARGS+=(
    -e "DISPLAY=${DISPLAY}"
    -e "QT_X11_NO_MITSHM=1"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
  )
fi

if [ "$#" -eq 0 ]; then
  docker run "${DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash
else
  docker run "${DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$@"
fi
