#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-curves-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=asr-curves-audit-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
"$HOME/.venvs/mls-rag-pilot/bin/python" rag/extensions/audit_asr_curves.py \
    --root "$PWD" --run "$PWD/rag/results/asr-matched-curves-20261117" \
    --plan "$PWD/rag/extensions/asr-curve-plan.json"
