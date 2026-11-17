#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-error-report
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=asr-error-report-%j.out
set -euo pipefail
umask 077
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cache=/home/s2487016/.cache/mls-asr-e2e-error
build="$cache/report-build-$SLURM_JOB_ID"
mkdir -p "$build/figures" "$build/qa"
for name in references.bib answer-analysis.tex answer-table.tex iclr2027_conference.sty iclr2027_conference.bst natbib.sty fancyhdr.sty math_commands.tex ICLR_2027_STYLE_SOURCE.txt README.md; do
    cp "$root/report/draft/$name" "$build/$name"
done
cp "$cache/report-candidate.tex" "$build/report.tex"
cp "$cache/draft-README.md" "$build/README.md"
cp "$cache/asr-e2e-error-results.md" "$build/asr-e2e-error-results.md"
cp "$root/report/asr-e2e-error-protocol.md" "$build/asr-e2e-error-protocol.md"
for name in uncapped-report-frontiers.pdf answer-cost-score.pdf uncapped-full-range.pdf; do
    cp "$root/report/draft/figures/$name" "$build/figures/$name"
done
cd "$build"
pdflatex -interaction=nonstopmode -halt-on-error report.tex > latex-first.log
bibtex report > bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex > latex-second.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex > latex-final.log
py="$HOME/.venvs/mls-rag-pilot/bin/python"
"$py" "$root/report/format_checker/check_report_format.py" report.pdf --json format-check.json
pdfinfo report.pdf > pdfinfo.txt
pdftoppm -r 110 -png report.pdf qa/page
"$py" - <<'PY'
from pathlib import Path
import zipfile
from PIL import Image, ImageOps, ImageDraw
files = sorted(Path('qa').glob('page-*.png'))
thumbs = []
for path in files:
    with Image.open(path) as source:
        thumb = ImageOps.contain(source.convert('RGB'), (306, 396))
        tile = Image.new('RGB', (326, 424), 'white')
        tile.paste(thumb, ((326-thumb.width)//2, 20))
        ImageDraw.Draw(tile).text((12, 405), path.stem, fill='black')
        thumbs.append(tile)
columns = 3
canvas = Image.new('RGB', (columns*326, ((len(thumbs)+columns-1)//columns)*424), '#dddddd')
for index, tile in enumerate(thumbs):
    canvas.paste(tile, ((index%columns)*326, (index//columns)*424))
canvas.save('qa/contact-sheet.png')
names = ['report.tex','references.bib','answer-analysis.tex','answer-table.tex','iclr2027_conference.sty','iclr2027_conference.bst','natbib.sty','fancyhdr.sty','math_commands.tex','README.md','ICLR_2027_STYLE_SOURCE.txt']
names += [str(p) for p in Path('figures').glob('*.pdf')]
names += ['asr-e2e-error-results.md','asr-e2e-error-protocol.md']
with zipfile.ZipFile('report-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as package:
    for name in names:
        package.write(name, name)
print('PDF pages rendered:', len(files))
PY
sha256sum report.tex report.pdf report-source.zip format-check.json
printf '%s\n' "$build" > "$cache/report-build-path.txt"
