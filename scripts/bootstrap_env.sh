#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 \$CONDA_PREFIX" >&2
  exit 2
fi

ENV_PREFIX="$1"
PYTHON="$ENV_PREFIX/bin/python"
CUDA_HOME="$ENV_PREFIX"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Compile FlashAttention for Blackwell rather than relying on the host's
# architecture auto-detection.
export TORCH_CUDA_ARCH_LIST="12.0"

if [[ ! -x "$PYTHON" ]]; then
  echo "No Python executable at $PYTHON. Create the environment from environment.yml first." >&2
  exit 2
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "No nvcc at $CUDA_HOME/bin/nvcc. Recreate the environment from environment.yml." >&2
  exit 2
fi

"$PYTHON" -m pip install --upgrade pip

# Install the runtime first: packages such as accelerate otherwise resolve the
# newest generic PyTorch from PyPI before the Blackwell-compatible build below.
"$PYTHON" -m pip install --upgrade --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128

# Install every upstream requirement except the runtime pair replaced below.
# The upstream pins target PyTorch 2.6, whose published CUDA builds predate
# Blackwell; retaining them would silently replace the CUDA 12.8 runtime.
sed -e '/^torch==/d' -e '/^torchvision==/d' third_party/LaWAM/requirements.txt \
  | "$PYTHON" -m pip install -r /dev/stdin

# Build against the environment's CUDA 12.8 compiler and installed PyTorch.
"$PYTHON" -m pip install flash-attn==2.8.3 --no-build-isolation
"$PYTHON" -m pip install --no-build-isolation -e third_party/LaWAM
"$PYTHON" -m pip install --no-build-isolation -e .[dev]
