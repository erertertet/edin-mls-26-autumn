# End-to-end ASR kernel numerical discrepancy

Measure the complete supplied GLM-ASR-Nano pipeline with custom CUDA kernel groups against the same model's all-PyTorch reference. This is a kernel-integration check on the supplied **one 3.50-second `test_audio.wav` clip**, not a new four-ASR experiment, a WER benchmark or a population error bound. Preserve all existing model/kernel source, fixed transcripts and retrieval results.

## Fixed comparison

- Use snapshot `61ba4e0b3309b6656edea3e93e419f7bd5c61957`, the existing float32 loader, one loaded model and one prepared processor input shared by every mode.
- Run the existing greedy `GlmAsr.generate` path with `none`, `task1,task2`, `task1,task2,task3` and `all`. As in the supplied adapter, partial CUDA modes also include the provided kernels. Repeat `none` and `all` once to expose within-run repeat discrepancies. No external service calls or new ASR-model comparison is included.
- Use PyTorch 2.10's **new precision controls only**: `torch.backends.fp32_precision`, `torch.backends.cuda.matmul.fp32_precision` and `torch.backends.cudnn.fp32_precision` set to `ieee`; record the actual settings. Do not mix these controls with deprecated `allow_tf32` settings. This separately instrumented numerical run does not replace the old uninstrumented timing measurements.
- Wrap the existing final-vocabulary `ops.softmax` call to retain the full pre-softmax logits and post-softmax probabilities at every generated position; retain projected audio embeddings too. Record token IDs and decoded output exactly. Restore instrumentation hooks after each run. Do not modify `model.py` or `ops.py`.
- Compare numerical arrays only where decoder-input prefixes match. The step that emits a first different greedy token still has a common input prefix; later divergent trajectories must not be described as same-input numerical error. Report aligned coverage and token/string disagreement separately. If all token IDs match, all decoding positions are aligned.

## Metrics and scope

For flattened candidate values `x` and PyTorch reference values `y`, compute in float64:

- maximum absolute discrepancy: `max(abs(x-y))`;
- mean absolute discrepancy: `mean(abs(x-y))`;
- RMS discrepancy: `sqrt(mean((x-y)**2))`;
- relative L2 discrepancy: `norm(x-y) / norm(y)` with an explicit zero-reference convention.

Report global full-vocabulary and per-position metrics for logits and probabilities, and the projected-audio metrics. Full coverage requires every vocabulary element at every aligned decoding position. Reject missing/non-finite traces; do not label a merely nonzero discrepancy as a failed test or tune a numerical acceptance tolerance after observing results. Also report token-ID equality and final decoded-text equality. Zero transcript disagreement does not imply zero floating-point error.

The historical benchmark's `Accuracy=100%` is the fraction of distinct expected words present after punctuation removal and uppercasing; it ignores order, repetitions and extra words. It must not be relabelled as WER=0. Existing single-kernel maximum errors are separate operator-level evidence, not an end-to-end error measurement.

## Environment, reuse and validation

The actual `mls-asr` interpreter, requirements/constraints, installed packages and import paths are checked before execution: Python 3.11.18, PyTorch 2.10.0+cu128, Transformers 5.0.0, NumPy 2.4.6 and Safetensors 0.8.0. Reuse the course model/processor and NumPy metrics; add no inference dependency or handwritten replacement model. Target-version official PyTorch documentation was checked for precision controls and floating-point comparison limits. A small runtime check validates the precision API inside the allocation before inference.

Build the unchanged default kernels in an isolated directory and run only on an Edinburgh Slurm H200 MIG GPU. Keep checkpoint/audio/source/library hashes, precision/device metadata, token IDs, traces and completion records. Independently recompute the saved-array discrepancy metrics and alignment accounting in a CPU allocation. Publish only aggregate correctness results and source; retain full numerical arrays in ignored results. Rebuild the report and run the supplied six-page formatting checker on a Slurm CPU node, then visually inspect the rendered pages.

References: [PyTorch 2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html), [PyTorch 2.10 precision controls](https://docs.pytorch.org/docs/2.10/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices), the supplied `hw1-asr/benchmark.py` and `glm_asr_cuda/model.py`/`ops.py`.
