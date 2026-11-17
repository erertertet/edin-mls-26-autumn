#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-curves-render
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=asr-curves-render-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$HOME/.cache/mls-asr-curves/matplotlib"
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
run="$PWD/rag/results/asr-matched-curves-20261117"
"$HOME/.venvs/mls-rag-pilot/bin/python" rag/extensions/render_asr_curve_analysis.py --run "$run" --output "$run/report.md"
