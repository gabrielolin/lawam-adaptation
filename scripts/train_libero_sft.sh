#!/usr/bin/env bash
# Project-owned launcher for LIBERO SFT with the local, gated DINOv3 weights.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAWAM_ROOT="${REPO_ROOT}/third_party/LaWAM"
LAM_YAML="${LAWAM_ROOT}/latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"

# Local runtime configuration; this is never written into tracked LaWAM files.
# Relative overrides are interpreted from the LaWAM root.
DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${LAWAM_ROOT}/weights/dinov3-vitb16-pretrain-lvd1689m}"
if [[ "${DINOV3_MODEL_PATH}" != /* ]]; then
  DINOV3_MODEL_PATH="${LAWAM_ROOT}/${DINOV3_MODEL_PATH}"
fi

if [[ ! -f "${DINOV3_MODEL_PATH}/config.json" || ! -f "${DINOV3_MODEL_PATH}/model.safetensors" ]]; then
  echo "DINOv3 files not found in: ${DINOV3_MODEL_PATH}" >&2
  echo "Set DINOV3_MODEL_PATH to the directory containing config.json and model.safetensors." >&2
  exit 2
fi
if ! grep -q '^  vision_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m$' "${LAM_YAML}"; then
  echo "Expected upstream DINOv3 model ID was not found in ${LAM_YAML}; refusing to rewrite an unexpected config." >&2
  exit 2
fi

# Escape sed replacement metacharacters in unusual filesystem paths.
sed_path="${DINOV3_MODEL_PATH//\\/\\\\}"
sed_path="${sed_path//&/\\&}"
sed_path="${sed_path//|/\\|}"
tmp_lam_yaml="$(mktemp "${TMPDIR:-/tmp}/lawam-lam-config.XXXXXX.yaml")"
trap 'rm -f "${tmp_lam_yaml}"' EXIT
sed "s|^  vision_model_id: facebook\/dinov3-vitb16-pretrain-lvd1689m$|  vision_model_id: ${sed_path}|" \
  "${LAM_YAML}" > "${tmp_lam_yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export LAWAM_ADAMW_FOREACH="${LAWAM_ADAMW_FOREACH:-false}"
export LAWAM_LAM_DECODER_GRAD_CHECKPOINTING="${LAWAM_LAM_DECODER_GRAD_CHECKPOINTING:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-256}"
VLA_NUM_WORKERS="${VLA_NUM_WORKERS:-0}"
VLA_VAL_NUM_WORKERS="${VLA_VAL_NUM_WORKERS:-0}"

cd "${LAWAM_ROOT}"
bash train_lawam.sh \
  --run_id=libero_sft_1gpu \
  --framework.action_model.lam_yaml_path="${tmp_lam_yaml}" \
  --datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers="${VLA_NUM_WORKERS}" \
  --datasets.vla_data.val_num_workers="${VLA_VAL_NUM_WORKERS}" \
  --trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  "$@"
