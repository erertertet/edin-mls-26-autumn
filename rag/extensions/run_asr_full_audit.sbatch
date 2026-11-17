#!/usr/bin/env bash
#SBATCH --job-name=mls-asr4-full-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
scratch=$(mktemp -d "/disk/scratch/mls-asr4-fullaudit-$USER-$SLURM_JOB_ID-XXXXXX")
trap 'case "$scratch" in /disk/scratch/mls-asr4-fullaudit-"$USER"-"$SLURM_JOB_ID"-*) rm -rf -- "$scratch";; esac' EXIT
export TMPDIR="$scratch/tmp" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
root="$HOME/edin-mls-asr-pilot"
run="$root/rag/results/asr-four-model-full-20261117"
state="$HOME/.cache/mls-asr-full"
python="$HOME/.venvs/mls-rag-pilot/bin/python"
cp -aL "$root/rag/data/rag_corpus" "$scratch/corpus"
cp -aL "$root/rag/models/enc_a" "$scratch/enc_a"
tokenizer=$("$python" -c 'import json,pathlib;print(json.loads((pathlib.Path.home()/".cache/mls-asr-pilot/tokenizer-metadata.json").read_text())["path"])')
cp "$tokenizer" "$scratch/tokenizer.json"
"$python" "$state/audit_asr_full.py" --root "$root" --run "$run" --data-dir "$scratch/corpus" \
  --encoder-dir "$scratch/enc_a" --tokenizer "$scratch/tokenizer.json"
"$python" "$state/refresh_asr_full_billing.py" "$run"
"$python" "$state/render_asr_full_results.py" --run "$run" --output "$run/report.md"
