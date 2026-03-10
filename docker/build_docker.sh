#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${1:-amd64}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE_NAME="${IMAGE_NAME:-interactive-world-sim}"
IMAGE_TAG="${IMAGE_TAG:-local}"

docker build \
  --platform "linux/${PLATFORM}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${REPO_DIR}/docker/Dockerfile" \
  "${REPO_DIR}"
