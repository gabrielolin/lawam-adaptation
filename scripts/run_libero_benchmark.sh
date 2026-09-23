#!/usr/bin/env bash
# Project-owned launcher for the vendored LaWAM LIBERO evaluation.
set -euo pipefail

if [[ -z "${LIBERO_HOME:-}" ]]; then
  echo "Set LIBERO_HOME to your external LIBERO checkout." >&2
  exit 2
fi
if [[ -z "${LIBERO_PYTHON:-}" ]]; then
  echo "Set LIBERO_PYTHON to the Python executable in the LIBERO environment." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-$(command -v python)}"

# Conda exports HOST=x86_64-conda-linux-gnu for its compiler toolchain. LaWAM's
# launcher interprets HOST as a network hostname, so always provide its policy
# server a real loopback address without changing the vendored launcher.
export HOST="${LAWAM_SERVER_HOST:-127.0.0.1}"

# LIBERO's packaged initial states are trusted local benchmark assets, not
# model weights. PyTorch 2.6 changed torch.load's default to weights_only=True,
# which prevents current LIBERO releases from opening those assets. Scope the
# legacy behavior to this launcher rather than modifying either upstream tree.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

cd "${REPO_ROOT}/third_party/LaWAM"
exec bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$@"
