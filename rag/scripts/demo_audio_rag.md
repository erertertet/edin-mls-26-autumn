# Live audio RAG demonstration

For a normal end-to-end demo with your own recording, no preset or reference is needed:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py "C:\path\question.mp3"
```

This runs ASR, retrieval and answer generation for all three policies. Gemini
scoring is optional and runs only when you supply a preset, query ID or independent
reference evidence. These references support evaluation, not answer generation.

Run `demo_audio_rag.py` on your laptop. It uploads the recording and demo worker
over SSH to Fuji3, converts audio there with FFmpeg, then runs all three policies
in order: **top-2**, **absolute similarity >= 0.8925**, and **gap from the top
similarity <= 0.009**. These are the dev macro-F1 winners.

The local launcher uses only Python's standard library and OpenSSH (`ssh` and
`scp`). It performs no local inference. Fuji3 uses the existing
`~/ragexp/activate.sh`, repository, models, CUDA library and private API key.
The script does not copy or print the key. Default ASR uses all custom CUDA
kernel groups. No Slurm submission is needed for this explicitly requested
Fuji3 demo.

## Record this question

> Are there any recommended software tools for value investing strategies?

Save your recording as an MP3 and, from the repository root, run:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py "C:\path\question.mp3" --preset value-investing
```

The preset supplies the matching practice-query relevance labels **only to the
judge**. Retrieval uses the actual freshly transcribed recording. Read the
question as written so these labels remain appropriate. The output prints the
transcript for inspection.

To first try the original supplied recording, omit the audio path:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py --preset value-investing
```

In the archived three-policy experiment, this question produced:

| Policy | Documents | Gemini score |
| --- | ---: | ---: |
| Top-2 | 2 | 1.000 |
| Absolute 0.8925 | 1 | 0.000 |
| Relative 0.009 | 3 | 0.500 |

The absolute cutoff retained a document about a value-investing book. Top-2
also included a document mentioning Stockdance. Relative retrieval added a
third document discussing a custom screening tool. This is an illustrative
example deliberately selected for differences, not an aggregate performance
comparison. A fresh voice recording, transcript, model response or judgement
can change these results. The script always shows the actual fresh output.

## What you will see

1. Audio upload and conversion to 16 kHz mono PCM16 WAV on Fuji3.
2. Model loading once, followed by three separate passes through ASR and query encoding.
3. For each policy: its transcript, document count, cosine scores, record IDs
   and **full retrieved text**. Cutoffs use the entire corpus, with top-1 fallback.
4. The complete Mistral Small 4 answer, then Gemini 2.5 Flash's correctness,
   grounding and completeness scores (0-4 each), explanation and total (sum/12).
5. A final comparison of document counts, full prompt tokens, answer latency,
   scores and reported API cost. Latency includes warm ASR through generation,
   excluding initial model loading, upload, conversion and judging.

Each policy has a separate generation and judgement. Gemini sees the same
independent reference documents for all policies and does not see policy names,
retrieved IDs or similarity scores.

Logs save locally under `rag/results/live-demo-<timestamp>/` and remotely under
`~/ragexp/live-demos/<timestamp>/`:

- `console.log`: readable step-by-step terminal transcript.
- `events.jsonl`: full retrieved text, responses, timings and API usage.
- `results.json`: structured policy comparison, including partial answers if judging fails.
- `config.json`: settings, without credentials.

This is a live run, not a replay. For failures, keep the printed remote directory.
Interrupting SSH can leave an in-flight API request running. Do not immediately
repeat a failed or interrupted run without checking its log.

## Your own questions

For another benchmark question, use its query ID and correct split:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py "question.mp3" --query-id q_EXAMPLE --split practice
```

For optional Gemini scoring of a new question, pass independent evidence inline:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py "question.mp3" --gold-text "Independent reference evidence goes here."
```

Or put the evidence in a UTF-8 file:

```powershell
py -3.13 rag/scripts/demo_audio_rag.py "question.mp3" --gold-file "reference.txt"
```

Without benchmark labels or reference evidence, the default still generates
all three answers and skips judging. Use `--retrieval-only` only for an
ASR/retrieval preview with no API calls. The script does not treat each policy's
retrieved documents as its own correctness reference.

## Options

- `--host wgfuji3`: use another configured SSH alias if direct `fuji3` access fails.
- `--gpu 1`: explicitly select a remote GPU. Otherwise use `activate.sh`.
- `--cuda none`: explicitly use the PyTorch ASR reference, if desired.
- `--budget 0.50`: default USD guard for generation plus judging. It checks live
  OpenRouter pricing and conservative prompt/output reservations before calls.
  This is an application guard, not a provider-enforced account limit.
- `--out "rag/results/my-live-demo"`: choose a new local output directory.
- `--remote-repo /home/zcm/ragexp/edin-mls-26-autumn`: override the remote checkout.
- `--remote-activate /home/zcm/ragexp/activate.sh`: override the remote environment.

Audio may be MP3, WAV, or another FFmpeg-readable format, up to 120 seconds.
The local source recording and existing experiment outputs remain unchanged.
Model defaults match the report: `mistralai/mistral-small-2603` and
`google/gemini-2.5-flash` through OpenRouter. The worker uses the private remote
environment or `rag/.env`. No new model installation is required.

## Verified live run

On 17 November 2026, the full upload path was checked with an MP3 encoded from
the supplied example recording. All three fresh CUDA ASR passes returned the
question above. Top-2, absolute and relative retrieved 2, 1 and 3 documents and
received Gemini scores of 1.0, 0.0 and 0.5, respectively. Six API calls cost
approximately $0.00201. Logs are in `rag/results/live-demo-mp3-validation/`.
An earlier fresh WAV run scored 1.0, 0.0 and 1.0, illustrating judge/answer
variability even when the transcript and retrieved sets match.

Six CPU-only unit tests passed on Fuji3, covering inclusive cutoff boundaries,
top-1 fallback, top-score-relative gaps, uncapped selections, invalid inputs
and SSH hostname validation. No local model inference was used.
