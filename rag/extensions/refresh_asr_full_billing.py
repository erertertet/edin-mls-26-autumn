"""Read-only post-run usage snapshot in Slurm; no paid request."""
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

if not os.getenv("SLURM_JOB_ID") or not socket.getfqdn().endswith(".inf.ed.ac.uk"):
    raise SystemExit("Run in Edinburgh Slurm")
os.umask(0o077)
run = Path(sys.argv[1])
load_dotenv(Path.home() / ".config/edin-mls/openrouter.env", override=True)
with OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1",
            max_retries=0, timeout=60) as client:
    current = client.get("/key", cast_to=dict)["data"]
start = json.loads((run / "billing-start.json").read_text())
billing = json.loads((run / "billing.json").read_text())
value = {"utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
         "key_usage_usd": current.get("usage"), "key_usage_before_usd": start["usage_usd"],
         "key_usage_delta_usd": current["usage"] - start["usage_usd"],
         "new_response_cost_sum_usd": billing["new_response_cost_sum_usd"],
         "all_successful_response_cost_sum_usd": billing["all_successful_response_cost_sum_usd"],
         "delta_minus_new_response_cost_usd": current["usage"] - start["usage_usd"] - billing["new_response_cost_sum_usd"]}
(run / "key-usage-after.json").write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value), flush=True)
