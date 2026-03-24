#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-sentiment-api}
echo "Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" .
