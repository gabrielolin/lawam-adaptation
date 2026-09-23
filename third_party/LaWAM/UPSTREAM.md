# LaWAM upstream provenance

- Upstream repository: https://github.com/RLinf/LaWAM.git
- Local Git remote: `lawam-upstream`
- Imported ref: `main`
- Imported upstream commit: `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`
- Import method: `git subtree add --prefix=third_party/LaWAM lawam-upstream main --squash`
- Import date: 2026-09-21

`third_party/LaWAM` is vendored third-party research code. Avoid modifying it
for project-specific research; put integrations in `src/lawam_adaptation`
whenever a wrapper or adapter can provide the needed hook.

## Local project patch

`starVLA/training/train_starvla.py` has two small launcher hooks not present
in upstream commit `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`:

- W&B initialization reads `WANDB_MODE`, defaulting to `offline` as upstream
  did. The project launcher opts into `online` by default.
- AdamW's `foreach` selection reads `LAWAM_ADAMW_FOREACH`, defaulting to
  PyTorch's automatic behavior (`auto`). The project launcher sets it to
  `false` to avoid foreach temporary-tensor peaks on a 16 GiB GPU.

`latent_action_model/core/utils/lam_decoder.py` also calls
`MultiheadAttention(..., need_weights=False)` because the decoder discards
attention weights; this permits PyTorch's memory-efficient scaled-dot-product
attention path and avoids materializing an unused attention matrix. It supports
optional per-layer activation checkpointing through
`LAWAM_LAM_DECODER_GRAD_CHECKPOINTING`; the project launcher enables it to
reduce decoder activation memory at the cost of recomputation.

The project launcher also defaults `PYTORCH_CUDA_ALLOC_CONF` to
`expandable_segments:True`; this is runtime configuration, not an upstream
source change.

Update this file if a future upstream subtree import changes the revision;
recheck whether the local patch still applies and remains necessary.

To update the subtree intentionally:

```bash
git fetch lawam-upstream main
git subtree pull --prefix=third_party/LaWAM lawam-upstream main --squash
```
