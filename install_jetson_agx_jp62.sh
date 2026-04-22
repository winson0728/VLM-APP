#!/usr/bin/env bash
set -euo pipefail

# Jetson AGX Orin + JetPack 6.2 CUDA/OpenCV setup script
# This script builds OpenCV with CUDA support and installs project deps.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-8.7}"
BUILD_DIR="${BUILD_DIR:-$HOME/opencv-build}"
SRC_DIR="${SRC_DIR:-$HOME/opencv-src}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

echo "[1/7] Checking JetPack packages..."
if ! dpkg -l | grep -q "nvidia-jetpack"; then
  echo "JetPack not detected. Installing nvidia-jetpack..."
  sudo apt-get update
  sudo apt-get install -y nvidia-jetpack
fi

echo "[2/7] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev libxvidcore-dev libx264-dev \
  libgtk-3-dev \
  libatlas-base-dev gfortran \
  libtbb2 libtbb-dev \
  libopenblas-dev liblapack-dev libeigen3-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  python3-dev python3-numpy python3-pip

echo "[3/7] Fetching OpenCV ${OPENCV_VERSION} sources..."
mkdir -p "${SRC_DIR}"
if [ ! -d "${SRC_DIR}/opencv" ]; then
  git clone --branch "${OPENCV_VERSION}" --depth 1 https://github.com/opencv/opencv.git "${SRC_DIR}/opencv"
else
  echo "OpenCV repo exists. Skipping clone."
fi

if [ ! -d "${SRC_DIR}/opencv_contrib" ]; then
  git clone --branch "${OPENCV_VERSION}" --depth 1 https://github.com/opencv/opencv_contrib.git "${SRC_DIR}/opencv_contrib"
else
  echo "OpenCV contrib repo exists. Skipping clone."
fi

echo "[4/7] Configuring OpenCV with CUDA..."
mkdir -p "${BUILD_DIR}"
cmake -S "${SRC_DIR}/opencv" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DOPENCV_EXTRA_MODULES_PATH="${SRC_DIR}/opencv_contrib/modules" \
  -DWITH_CUDA=ON \
  -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  -DWITH_CUDNN=ON \
  -DOPENCV_DNN_CUDA=ON \
  -DWITH_CUBLAS=ON \
  -DENABLE_FAST_MATH=ON \
  -DCUDA_FAST_MATH=ON \
  -DBUILD_opencv_python3=ON \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF

echo "[5/7] Building OpenCV (this may take a while)..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[6/7] Installing OpenCV..."
sudo cmake --build "${BUILD_DIR}" --target install
sudo ldconfig

echo "[7/7] Installing project dependencies (excluding opencv-python)..."
REQS_JETSON="${PROJECT_DIR}/requirements-jetson.txt"
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  # Create a Jetson-friendly requirements file without opencv-python
  grep -v "^opencv-python==" "${PROJECT_DIR}/requirements.txt" > "${REQS_JETSON}"
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "${REQS_JETSON}"
else
  echo "requirements.txt not found in project directory."
fi

echo "Done."
echo "Verify CUDA in OpenCV with:"
echo "  python3 - <<'PY'"
echo "  import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
echo "  PY"
