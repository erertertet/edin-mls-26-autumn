"""Targeted scientific aggregation and SDK-transport checks; no network calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx2
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
import asr_retrieval_pilot as pilot


class PilotChecks(unittest.TestCase):
    def test_native_json_transport_preserves_usage_and_headers(self):
        seen = []

        def respond(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx2.Response(200, json={"text": "Synthetic transcript", "usage": {"cost": 0.0001}}, headers={"x-request-id": "synthetic-request"})

        with httpx2.Client(transport=httpx2.MockTransport(respond)) as transport:
            with OpenAI(api_key="synthetic-not-a-real-key", base_url="https://openrouter.ai/api/v1", http_client=transport, max_retries=0) as client:
                response = client.post("/audio/transcriptions", cast_to=httpx2.Response, body={"model": pilot.MODELS["mai"], "input_audio": {"data": "synthetic", "format": "wav"}, "provider": {"options": {"azure": {"diarization": {"enabled": False}}}}})
        self.assertEqual(seen[0][0], "https://openrouter.ai/api/v1/audio/transcriptions")
        self.assertFalse(seen[0][1]["provider"]["options"]["azure"]["diarization"]["enabled"])
        self.assertEqual(response.headers["x-request-id"], "synthetic-request")
        self.assertEqual(response.json()["usage"]["cost"], 0.0001)

    def test_all_settings_use_the_same_complete_query_intersection(self):
        rows = []
        for q, index in [("a", 0), ("b", 1), ("excluded", 2)]:
            for arm in pilot.MODELS:
                for rule in pilot.RULES:
                    if (q, arm, rule) == ("excluded", "voxtral", "cutoff"):
                        continue
                    recalls = [.5, .5, 0] if arm == "glm" else [.75, .25, 1]
                    tokens = [100, 200, 1000] if arm == "glm" else [80, 160, 1]
                    rows.append(dict(query_id=q, arm=arm, rule=rule, recall=recalls[index], context_tokens=tokens[index], documents=5, fallback=False))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            self.assertFalse(pilot.summarize(rows, ["a", "b", "excluded"], out, [], {}))
            result = json.loads((out / "summary.json").read_text())
        self.assertEqual(result["complete_paired_queries"], ["a", "b"])
        self.assertEqual({r["paired_queries"] for r in result["table"]}, {2})
        for row in result["table"]:
            self.assertEqual(row["mean_recall"], .5)
            self.assertEqual(row["delta_recall_pp_vs_glm"], 0)
            self.assertEqual(row["delta_context_tokens_vs_glm"], 0 if row["arm"] == "glm" else -30)


if __name__ == "__main__":
    unittest.main()
