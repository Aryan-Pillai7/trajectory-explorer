# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - not yet released

The first version. Tested end to end on Pythia-70M.

### Added

- `diff A B`: compares two checkpoints of the same architecture, tensor by tensor, from the
  weights alone, and writes one self-contained HTML report (inline SVG, no JavaScript, works
  offline). For each tensor it reports the relative and absolute change, and for matrices the
  effective rank and r90 of the change, with low-rank/dense and concentrated/spread labels.
  Tensors are grouped into a layer × component heatmap with separate panels for weight matrices
  and vectors (biases, norm scales). `--control A2:B2` adds a second pair as a reference scale.
  `--json` writes the full result.
- `trajectory`: diffs every adjacent pair of an ordered list of checkpoints, or of a repo's
  `stepN` branches (`--steps default|all|N,N,...`; the default is about 25 steps spaced on a log
  scale). One report with an interval heatmap, a line chart per component, an interval table,
  and a note on interval lengths.
- A noise floor based on storage rounding, `3 × √(u_A² + u_B²) / √3`. Changes at or below it are
  muted and never counted as significant. Identical files report "No significant difference".
- The effective-precision rule: a float32 tensor whose values are all exact float16 numbers in
  both checkpoints gets the float16 floor. This matters for Pythia, whose float32 step files
  hold float16 values.
- Tensors that start at exactly zero are reported separately ("0→"), since they have no
  relative change.
- Checkpoints from local `.safetensors` files or from the Hugging Face Hub (`org/name@revision`),
  using the standard library only. Downloads resume after an interruption, are checked against
  the Hub's sha256, and use an optional `HF_TOKEN` that is never sent to another host or logged.
- A rolling checkpoint store: at most two downloaded checkpoints on disk by default
  (`--max-checkpoints`). Files in use are never evicted, the least recently used one goes
  first, and an old file is removed only after its replacement has downloaded successfully.
- A download guard: every command prints what it will download first. Over 2 GB it measures
  throughput on the first file, prints an estimate and asks, or stops without a terminal unless
  `--yes` is given.
- A metrics cache: per-pair measurements are saved as JSON keyed by the files' sha256, so
  reports regenerate without downloading or measuring again.
- Its own safetensors reader: one tensor at a time as float32, with F32, F16 and BF16 supported.
  Peak memory on Pythia-70M was 433 MB.
- Docker-only setup: a runtime image with numpy as its only dependency, and compose services
  `te`, `test`, `lint` and `format`. All generated data goes to the `TE_DATA_DIR` folder.
