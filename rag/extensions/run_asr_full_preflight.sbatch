#!/usr/bin/env bash
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
"$HOME/.venvs/mls-rag-pilot/bin/python" "$HOME/.cache/mls-asr-full/asr_full_preflight.py"
