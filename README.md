# LaWAM Adaptation

Research code for studying adaptations of [LaWAM](https://github.com/RLinf/LaWAM). The upstream implementation is vendored unchanged in `third_party/LaWAM`; project-specific adapters, benchmarks, and experiments belong in `src/lawam_adaptation` and `experiments`.

## Installation

This project targets Python 3.10 and NVIDIA Blackwell GPUs (including RTX 5080) using PyTorch CUDA 12.8. Create the environment and install both packages:

```bash
conda env create --prefix .conda-env --file environment.yml
bash scripts/bootstrap_env.sh .conda-env
conda activate "$PWD/.conda-env"
```

`bootstrap_env.sh` installs LaWAM's upstream requirements, replaces only the PyTorch runtime with the CUDA 12.8 build required for Blackwell, compiles a matching FlashAttention release, then installs LaWAM and this project in editable mode. It does not download datasets or checkpoints. Configure their locations through environment variables or experiment configuration, never source code.

## Verification

```bash
python scripts/smoke_test.py
pytest
```

The smoke test reports the PyTorch/CUDA version, detects the GPU, and imports `starVLA` (LaWAM) plus `lawam_adaptation`.

## Upstream provenance

LaWAM is imported as a squashed Git subtree from the `lawam-upstream` remote. The exact upstream revision and update instructions are recorded in [third_party/LaWAM/UPSTREAM.md](third_party/LaWAM/UPSTREAM.md).

## Repository structure

```text
src/lawam_adaptation/   Project-owned adapters and reusable research code
experiments/            Thin experiment orchestration and configurations
tests/                  Fast deterministic project tests
third_party/LaWAM/      Unmodified upstream LaWAM subtree
scripts/                Reproducible environment bootstrap and smoke test
```
