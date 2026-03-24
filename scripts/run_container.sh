#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-sentiment-api}
PORT=${2:-5000}

echo "Running container ${IMAGE} exposing port ${PORT} -> container:5000"
docker run --rm -p ${PORT}:5000 "${IMAGE}"
