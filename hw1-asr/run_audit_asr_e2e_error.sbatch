#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-e2e-error-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=asr-e2e-error-audit-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
"$HOME/.venvs/mls-rag-pilot/bin/python" hw1-asr/audit_asr_e2e_error.py \
    --root "$PWD" --run "$PWD/rag/results/asr-e2e-error-20261117"
