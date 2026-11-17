#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-full-e2e
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=asr-full-e2e-%A_%a.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mls-asr"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="/disk/scratch/mls-asr-full-e2e-$USER-20261117/hf"
export TMPDIR="/disk/scratch/mls-asr-full-e2e-$USER-20261117/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
export ASR_LIB="$PWD/rag/results/asr-full-e2e-20261117/libasr_kernels.so"
"$HOME/.conda/envs/mls-asr/bin/python" hw1-asr/measure_asr_full_e2e.py \
    --root "$PWD" --run "$PWD/rag/results/asr-full-e2e-20261117" --shard "$SLURM_ARRAY_TASK_ID"
