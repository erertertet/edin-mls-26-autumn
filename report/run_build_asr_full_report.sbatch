#!/usr/bin/env bash
# Submit from the repository root after audited full-audio prose has been added.
# Optional: REPORT_SOURCE_DIR, REPORT_TEX, REPORT_BUILD_CACHE, REPORT_RENDER_DPI.
#SBATCH --job-name=mls-asr-full-report
#SBATCH --partition=Teaching
#SBATCH --exclude=saxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=asr-full-report-%j.out
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
case "$(hostname -f)" in *.inf.ed.ac.uk) ;; *) exit 1;; esac
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
source_dir="${REPORT_SOURCE_DIR:-$root/report/draft}"
cache="${REPORT_BUILD_CACHE:-$HOME/.cache/mls-asr-full-e2e-report}"
build="$cache/report-build-$SLURM_JOB_ID"
py="$HOME/.venvs/mls-rag-pilot/bin/python"
test -f "$source_dir/report.tex"
test -x "$py"
test ! -e "$build/report.pdf"
mkdir -p "$build/qa"
export REPORT_BUILD_ROOT="$build" REPORT_SOURCE_ROOT="$source_dir" REPORT_REPO_ROOT="$root"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY

{
    date -u --iso-8601=seconds
    hostname -f
    git -C "$root" rev-parse HEAD
    for tool in pdflatex bibtex pdfinfo pdftoppm; do command -v "$tool"; done
    pdflatex --version
    bibtex --version
    pdfinfo -v
    pdftoppm -v
    "$py" -m pip list --format=freeze
} > "$build/environment.log" 2>&1

# Preserve the complete project source tree. This includes new \input modules
# such as asr-agreement.tex and any full-audio appendix file, not a fixed list.
"$py" - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
from PIL import Image, ImageOps
import fitz
import numpy as np

source, build = (Path(os.environ[key]).resolve() for key in ('REPORT_SOURCE_ROOT', 'REPORT_BUILD_ROOT'))
manifest = {}
for path in sorted(source.rglob('*')):
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    latex = path.suffix.lower() in {'.tex', '.bib', '.sty', '.bst'}
    figure = path.suffix.lower() == '.pdf' and relative.parts[0] == 'figures'
    support = relative.as_posix() in {'README.md', 'ICLR_2027_STYLE_SOURCE.txt'}
    if latex or figure or support:
        target = build / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Staged source differs: ' + relative.as_posix())
        manifest[relative.as_posix()] = {'sha256': digest, 'source': str(path)}
candidate = os.environ.get('REPORT_TEX')
if candidate:
    path = Path(candidate).resolve()
    shutil.copyfile(path, build / 'report.tex')
    manifest['report.tex'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source': str(path)}
required = ('report.tex', 'references.bib', 'iclr2027_conference.sty', 'iclr2027_conference.bst',
            'natbib.sty', 'fancyhdr.sty', 'math_commands.tex')
if any(not (build / name).is_file() for name in required):
    raise RuntimeError('Missing required course source/style input')
probe = ImageOps.contain(Image.new('RGB', (2, 2)), (1, 1))
if probe.size != (1, 1):
    raise RuntimeError('Installed Pillow thumbnail API probe failed')
(build / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(build / 'rendering-libraries.json').write_text(json.dumps({name: {
    'version': importlib.metadata.version(distribution), 'path': importlib.util.find_spec(name).origin}
    for name, distribution in [('PIL', 'Pillow'), ('fitz', 'PyMuPDF')]}, indent=2) + '\n')
(build / 'numpy-precision.json').write_text(json.dumps({'numpy': np.__version__,
    'float64_mantissa': int(np.finfo(np.float64).nmant),
    'longdouble_mantissa': int(np.finfo(np.longdouble).nmant),
    'longdouble_bytes': int(np.dtype(np.longdouble).itemsize)}) + '\n')
PY

cd "$build"
pdflatex -recorder -interaction=nonstopmode -halt-on-error report.tex > latex-first.log
bibtex report > bibtex.log
pdflatex -recorder -interaction=nonstopmode -halt-on-error report.tex > latex-second.log
pdflatex -recorder -interaction=nonstopmode -halt-on-error report.tex > latex-final.log
if "$py" - <<'PY'
from pathlib import Path
import re
log = Path('report.log').read_text(errors='replace')
raise SystemExit(0 if re.search(r'Label\(s\) may have changed|There were undefined references|Rerun to get cross-references right', log) else 1)
PY
then
    pdflatex -recorder -interaction=nonstopmode -halt-on-error report.tex > latex-extra.log
fi

# Render even if the supplied format checker finds a problem, so the complete
# page images remain available for diagnosis and visual review.
format_status=0
"$py" "$root/report/format_checker/check_report_format.py" report.pdf --json format-check.json \
    > format-check.log 2>&1 || format_status=$?
export REPORT_FORMAT_EXIT="$format_status"
pdfinfo report.pdf > pdfinfo.txt
pdftoppm -r "${REPORT_RENDER_DPI:-120}" -png report.pdf qa/page

"$py" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
from PIL import Image, ImageOps, ImageDraw

build = Path.cwd().resolve()
result = json.loads(Path('format-check.json').read_text())
boundary = result['metrics'].get('references_boundary', {})
main_pages = boundary.get('main_body_pages')
format_ok = (result['status'] != 'FAIL' and boundary.get('reliable')
             and main_pages is not None and main_pages <= 6
             and result['metrics'].get('configured_main_body_page_limit') == 6)
log = Path('report.log').read_text(errors='replace')
unresolved = [line for line in log.splitlines() if
              re.search(r'LaTeX Warning: (?:Citation|Reference).*undefined|There were undefined references|Label\(s\) may have changed|Rerun to get cross-references right', line)]
files = sorted(Path('qa').glob('page-*.png'), key=lambda p: int(p.stem.rsplit('-', 1)[1]))
if not files or len(files) != result['metrics']['pages_total']:
    raise RuntimeError('Rendered page count differs from compiled PDF')
tiles = []
for path in files:
    with Image.open(path) as source:
        thumbnail = ImageOps.contain(source.convert('RGB'), (306, 396))
        tile = Image.new('RGB', (326, 424), 'white')
        tile.paste(thumbnail, ((326 - thumbnail.width) // 2, 12))
        ImageDraw.Draw(tile).text((12, 405), path.stem, fill='black')
        tiles.append(tile)
sheet = Image.new('RGB', (3 * 326, ((len(tiles) + 2) // 3) * 424), '#dddddd')
for index, tile in enumerate(tiles):
    sheet.paste(tile, ((index % 3) * 326, (index // 3) * 424))
sheet.save('qa/contact-sheet.png')

# The TeX engine's recorder supplies actual dependencies. Stage all figure PDFs
# for compilation, but archive only PDF figures that this draft actually used.
project_inputs = set()
for line in Path('report.fls').read_text(errors='replace').splitlines():
    if not line.startswith('INPUT '):
        continue
    path = Path(line[6:]).resolve()
    try:
        relative = path.relative_to(build)
    except ValueError:
        continue
    if not path.is_file():
        raise RuntimeError('Recorded project dependency is missing: ' + str(relative))
    project_inputs.add(relative.as_posix())
manifest = json.loads(Path('input-manifest.json').read_text())
for relative, item in manifest.items():
    if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Build changed staged source/style: ' + relative)
source_names = {name for name in manifest if Path(name).suffix.lower() in {'.tex', '.bib', '.sty', '.bst'}}
source_names |= {name for name in manifest if name in {'README.md', 'ICLR_2027_STYLE_SOURCE.txt'}}
source_names |= {name for name in project_inputs if Path(name).suffix.lower() == '.pdf'}
with zipfile.ZipFile('report-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as package:
    for name in sorted(source_names):
        package.write(name, name)
Path('project-inputs.json').write_text(json.dumps(sorted(project_inputs), indent=2) + '\n')
qa = {'status': 'pass' if format_ok and not unresolved else 'fail', 'format_status': result['status'],
      'format_checker_exit': int(os.environ['REPORT_FORMAT_EXIT']), 'main_body_pages': main_pages,
      'total_pages': result['metrics']['pages_total'], 'rendered_pages': len(files),
      'undefined_reference_or_citation_warnings': unresolved, 'visual_review': 'pending',
      'overfull_box_messages': [line for line in log.splitlines() if 'Overfull ' in line],
      'source_archive_files': len(source_names)}
Path('build-qa.json').write_text(json.dumps(qa, indent=2) + '\n')
Path('artifact-sha256.json').write_text(json.dumps({name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for name in ['report.tex', 'report.pdf', 'report-source.zip', 'format-check.json', 'qa/contact-sheet.png']}, indent=2) + '\n')
print(json.dumps(qa), flush=True)
PY
printf '%s\n' "$build" > "$cache/report-build-path.txt"
sha256sum report.tex report.pdf report-source.zip format-check.json qa/contact-sheet.png
"$py" - <<'PY'
import json
from pathlib import Path
if json.loads(Path('build-qa.json').read_text())['status'] != 'pass':
    raise SystemExit('PDF retained for review; resolve format/page-limit/reference failures before delivery')
PY
