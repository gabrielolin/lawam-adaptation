# LaWAM Adaptation

Research code for studying adaptations of [LaWAM](https://github.com/RLinf/LaWAM). The upstream implementation is vendored unchanged in `third_party/LaWAM`; project-specific adapters, benchmarks, and experiments belong in `src/lawam_adaptation` and `experiments`.

## Installation

This project targets Python 3.10 and NVIDIA Blackwell GPUs (including RTX 5080) using PyTorch CUDA 12.8. Create the named Conda environment and install LaWAM plus this project:

```bash
conda env create --name lawam-adaptation --file environment.yml
conda activate lawam-adaptation
bash scripts/bootstrap_env.sh "$CONDA_PREFIX"
```

`bootstrap_env.sh` installs LaWAM's upstream requirements into `lawam-adaptation`, replaces only the PyTorch runtime with the CUDA 12.8 build required for Blackwell, compiles a matching FlashAttention release, then installs LaWAM and this project in editable mode. It does not download datasets or checkpoints. Configure their locations through environment variables or experiment configuration, never source code.

## Verification

```bash
pytest
```

For one-off CUDA diagnostics, use the ignored `scratch/` directory rather than adding permanent scripts to the repository.

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
