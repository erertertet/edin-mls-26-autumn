#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-edit-distance-report
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=asr-edit-distance-render-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
run="$PWD/rag/results/asr-edit-distance-20261117"
"$HOME/.venvs/mls-rag-pilot/bin/python" rag/extensions/render_asr_edit_distance.py --run "$run" --output "$run/report.md"
