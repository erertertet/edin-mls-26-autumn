#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-edit-distance-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=asr-edit-distance-audit-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
"$HOME/.venvs/mls-rag-pilot/bin/python" rag/extensions/audit_asr_edit_distance.py \
    --root "$PWD" --full-run "$PWD/rag/results/asr-four-model-full-20261117" \
    --run "$PWD/rag/results/asr-edit-distance-20261117"
