#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${ROOT_DIR}/datasets/mnist.zip"
DATA_DIR="${ROOT_DIR}/datasets/mnist"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Missing dataset archive: ${ZIP_PATH}"
  exit 1
fi

mkdir -p "${DATA_DIR}"

if [[ -f "${DATA_DIR}/mnist_train.csv" && -f "${DATA_DIR}/mnist_test.csv" ]]; then
  echo "MNIST CSV files already present in ${DATA_DIR}"
  exit 0
fi

echo "Extracting MNIST dataset to ${DATA_DIR}..."
unzip -o "${ZIP_PATH}" -d "${DATA_DIR}"

echo "Done. Expected files:"
echo "  ${DATA_DIR}/mnist_train.csv"
echo "  ${DATA_DIR}/mnist_test.csv"
