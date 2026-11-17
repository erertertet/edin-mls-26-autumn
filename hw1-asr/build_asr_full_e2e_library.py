"""Record the unchanged kernel sources and the actual SM86 Slurm build bytes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path.cwd()
run = root / "rag/results/asr-full-e2e-a6000-20261117"
assert os.getenv("SLURM_JOB_ID") and socket.getfqdn() == "landonia11.inf.ed.ac.uk"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
sources = [root / "hw1-asr/Makefile"] + sorted(p for p in (root / "hw1-asr/csrc").iterdir() if p.suffix in (".cu", ".cuh", ".h"))
library = run / "build/sm_86/libasr_kernels.so"
proof = dict(status="complete", library_sha256=sha(library), sm=86,
             kernel_tests_log_sha256=sha(run / "kernel-tests.log"),
             nvcc_version=subprocess.check_output(["/opt/cuda-12.8.0/bin/nvcc", "--version"], text=True),
             compiler_version=subprocess.check_output(["/usr/bin/g++", "--version"], text=True),
             source_sha256={str(p.relative_to(root)): sha(p) for p in sources},
             slurm_job_id=os.environ["SLURM_JOB_ID"], hostname=socket.getfqdn())
(run / "library-build.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof), flush=True)
