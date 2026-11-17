#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-curves-analysis
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=asr-curves-analysis-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$HOME/.cache/mls-asr-curves/matplotlib"
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
python="$HOME/.venvs/mls-rag-pilot/bin/python"
"$python" -m unittest discover -s rag/tests -p test_analyze_asr_curves.py -v
run="$PWD/rag/results/asr-matched-curves-20261117"
"$python" rag/extensions/analyze_asr_curves.py --run "$run" --plan rag/extensions/asr-curve-plan.json
"$python" rag/extensions/render_asr_curve_analysis.py --run "$run" --output "$run/report.md"
