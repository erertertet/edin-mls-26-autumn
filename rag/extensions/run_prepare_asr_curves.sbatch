#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-curves-cache
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=asr-curves-cache-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
scratch=$(mktemp -d "/disk/scratch/mls-asr-curves-$USER-$SLURM_JOB_ID-XXXXXX")
trap 'case "$scratch" in /disk/scratch/mls-asr-curves-"$USER"-"$SLURM_JOB_ID"-*) rm -rf -- "$scratch";; esac' EXIT
export TMPDIR="$scratch/tmp" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
mkdir -p "$TMPDIR"
python="$HOME/.venvs/mls-rag-pilot/bin/python"
"$python" -m unittest discover -s rag/tests -p test_asr_curve_statistics.py -v
cp -aL rag/data/rag_corpus "$scratch/corpus"
tokenizer=$("$python" -c 'import json,pathlib;print(json.loads((pathlib.Path.home()/".cache/mls-asr-pilot/tokenizer-metadata.json").read_text())["path"])')
cp "$tokenizer" "$scratch/tokenizer.json"
"$python" rag/extensions/prepare_asr_curve_cache.py --root "$PWD" \
  --full-run "$PWD/rag/results/asr-four-model-full-20261117" \
  --out "$PWD/rag/results/asr-matched-curves-20261117" --data-dir "$scratch/corpus" \
  --tokenizer "$scratch/tokenizer.json" --plan "$PWD/rag/extensions/asr-curve-plan.json"
