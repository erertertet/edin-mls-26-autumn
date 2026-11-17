#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-full-e2e-prepare
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=asr-full-e2e-prepare-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
"$HOME/.venvs/mls-rag-pilot/bin/python" -m py_compile hw1-asr/prepare_asr_full_e2e.py hw1-asr/measure_asr_full_e2e.py
"$HOME/.venvs/mls-rag-pilot/bin/python" hw1-asr/prepare_asr_full_e2e.py \
    --root "$PWD" --run "$PWD/rag/results/asr-full-e2e-20261117" \
    --stage-dir "/disk/scratch/mls-asr-full-e2e-$USER-20261117"
