# LaWAM Adaptation

Research code for studying adaptations of [LaWAM](https://github.com/RLinf/LaWAM). The upstream implementation is vendored in `third_party/LaWAM`; project-specific adapters, benchmarks, and experiments belong in `src/lawam_adaptation` and `experiments`. One documented local patch makes W&B mode environment-configurable while retaining offline mode as LaWAM's default.

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

## LIBERO evaluation

Install LIBERO separately in its own Conda environment, then activate `lawam-adaptation` and set `LIBERO_HOME` and `LIBERO_PYTHON` to its external checkout and environment. The LIBERO environment must include the simulator stack and LaWAM's small evaluation-client dependencies (`tyro`, `websockets`, `msgpack`, `imageio`, and `matplotlib`); it does not need FlashAttention or the LaWAM model itself.

Run LaWAM's vendored benchmark through the project-owned adapter:

```bash
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero-environment/bin/python
scripts/run_libero_benchmark.sh "$CKPT_PATH"
```

The adapter forces the policy server to use `127.0.0.1`, avoiding Conda's compiler-toolchain `HOST` variable being misinterpreted as a network hostname. It also enables PyTorch's legacy loader only for LIBERO's trusted local task-state assets, which current LIBERO releases require with PyTorch 2.6+. Benchmark outputs remain in LaWAM's ignored `results/` directory.

## LIBERO SFT

The project-owned launcher points LaWAM at the local DINOv3 download without
editing the vendored LAM config:

```bash
scripts/train_libero_sft.sh
```

By default it uses one visible GPU, online W&B logging, per-device batch size 1, gradient
accumulation 256 (effective batch size 256), zero data-loader workers, and
non-foreach AdamW, LAM-decoder activation checkpointing, and expandable CUDA allocator segments for a lower-memory
starting point. The non-foreach optimizer path may be slower. Increase `PER_DEVICE_BATCH_SIZE` only if GPU
memory permits, adjusting `GRADIENT_ACCUMULATION_STEPS` so their product
remains 256. Set `CUDA_VISIBLE_DEVICES` to select a GPU, or
`DINOV3_MODEL_PATH` to point at a different local DINOv3 directory. Worker
counts can be increased with `VLA_NUM_WORKERS` and `VLA_VAL_NUM_WORKERS`.
Set `LAWAM_ADAMW_FOREACH=auto` to use PyTorch's default optimizer selection.
Additional LaWAM config overrides can be passed as arguments to the launcher.
Authenticate once with `wandb login`; logs go to the `finetune-LIBERO` project
unless overridden with `--wandb_project=...`. Set `WANDB_MODE=offline` to
disable online syncing for a run. LaWAM logs losses and learning-rate metrics
at its configured logging interval.

## Upstream provenance

LaWAM is imported as a squashed Git subtree from the `lawam-upstream` remote. The exact upstream revision and update instructions are recorded in [third_party/LaWAM/UPSTREAM.md](third_party/LaWAM/UPSTREAM.md).

## Repository structure

```text
src/lawam_adaptation/   Project-owned adapters and reusable research code
experiments/            Thin experiment orchestration and configurations
tests/                  Fast deterministic project tests
third_party/LaWAM/      Unmodified upstream LaWAM subtree
scripts/                Reproducible environment bootstrap and launchers
```
