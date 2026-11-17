#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-full-e2e-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=landonia11
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
python="$HOME/.venvs/mls-rag-pilot/bin/python"
run="$PWD/rag/results/asr-full-e2e-a6000-20261117"
"$python" -m py_compile hw1-asr/summarize_asr_full_e2e.py hw1-asr/audit_asr_full_e2e.py
"$python" hw1-asr/summarize_asr_full_e2e.py --root "$PWD" --run "$run"
"$python" hw1-asr/audit_asr_full_e2e.py --root "$PWD" --run "$run"
"$python" hw1-asr/summarize_asr_full_e2e.py --run "$run" --report-only
