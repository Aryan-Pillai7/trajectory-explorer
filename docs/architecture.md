# Architecture

This page is for someone opening the repository for the first time. It walks through the
two pipelines, one module at a time, and says what each piece takes in and hands on. All
code is in `src/trajectory_explorer/`. The only runtime dependency is numpy.

```
Measuring a pair
  reader ──> arch ──> metrics + noise ──> diff engine ──> report / trajectory_report
                                              │
                                              └──> metrics cache (JSON, keyed by sha256)

Getting the files
  CLI argument ──> sources ──> (hub) ──> store ──> local file
                       └────> CLI download plan + guard (before anything is fetched)
```

The two meet in the diff engine. It asks a *source* for a local file only when a pair really
has to be measured, that is, when the metrics cache has no entry for it.

## Measuring a pair

### reader.py: safetensors, one tensor at a time

A safetensors file is an 8-byte length, a JSON header (each tensor's name, dtype, shape and
byte offsets), then the raw bytes. `read_header` parses the header only, so listing a
checkpoint never touches weight data. `Checkpoint.load(name)` does a plain seek + read of one
tensor and returns it as float32. F32, F16 and BF16 are supported. BF16 is widened exactly, by
putting its 16 bits in the top half of a float32.

The reader is our own, not the `safetensors` library (which is a dev dependency, used only as a
reference in tests). Two reasons: the library's numpy backend rejects BF16, and memory-mapping
kept every tensor read so far resident until the file closed. With plain reads, peak memory on
Pythia-70M dropped from about 790 MB to about 400 MB.

### arch.py: where a tensor sits

`classify(name)` maps a tensor name to a `TensorKey(layer, component)`. The components are the
heatmap columns: `embed`, `attn_qkv`, `attn_out`, `mlp_in`, `mlp_out`, `norm`, `unembed`,
`other`. Rules exist for GPT-NeoX (Pythia) and Llama-style names (SmolLM2, Llama, Mistral,
Qwen2). Anything else falls back to a generic "layers.N." pattern plus keywords.

`parameter_specs` drops known buffers by name (attention masks, rotary `inv_freq`) and any
non-float tensor. `check_compatible` requires the same parameter names and shapes on both
sides (dtypes may differ) and otherwise raises `ArchitectureMismatch`, which lists what is
only in A, only in B, or differently shaped (exit code 3).

### metrics.py and noise.py: numbers for one tensor

`measure_tensor(a, b, dtype_a, dtype_b)` takes two float32 arrays and returns a
`TensorMeasurement`:

- norms of A and B and of the delta B − A;
- for matrices: the singular values of the delta, from the eigenvalues of the smaller Gram
  matrix (DᵀD or DDᵀ, accumulated in float64 over row chunks, so a 50304 × 512 embedding
  needs a 512 × 512 matrix, not a full SVD), then the entropy effective rank, r90 and the
  effective rank a Gaussian matrix of the same shape would have;
- the share of the delta's energy held by the top 5% of rows;
- whether each float32 side holds only float16-exact values.

Everything is accumulated in float64 over bounded chunks, so memory stays near the size of the
two tensors plus their difference.

`assess_tensor(measurement, control)` turns that into `TensorMetrics`. It picks the precision
each side really carries, computes the noise floor, sets a status (`no_change`, `from_zero`,
`below_floor`, `below_control`, `significant`) and applies the rank and concentration labels.
The floor formula, the thresholds and the labels are in `noise.py`, and
[metrics.md](metrics.md) explains them.

Measuring and assessing are separate on purpose. Only measurements are cached, so changing the
control pair or the floor rules never forces a re-measurement.

### diff.py: the engine

`diff_checkpoints(a, b, options)` does, in order:

1. **Identity.** Each source knows its sha256 before any download (the Hub's LFS hash, or a
   hash of the local file, computed once per file per run). Byte-identical files return
   straight away with "no difference" and nothing is read.
2. **Measure or load.** The metrics cache is `metrics/<shaA>_<shaB>_m<METRICS_VERSION>.json`.
   On a hit, nothing is fetched. On a miss, both files are fetched (and pinned in the store),
   each shared tensor is loaded one at a time from A and B, measured, and freed, and the
   result is saved to the cache.
3. **Control (optional).** A control pair is measured the same way, but only after the
   compared pair is finished and cached. With a two-file store, that means the compared pair's
   files never have to be downloaded twice.
4. **Assess and group.** Each tensor is assessed against its floor and its control value.
   Tensors are then grouped into cells by (layer, component, kind), where kind is `matrix`
   (2-D and up) or `vector` (biases, norm scales). Matrices and vectors are never mixed in one
   cell.

The output is a `DiffResult` with the two sources, per-tensor metrics and the groups. It
round-trips through JSON exactly (`to_json` / `from_json`), which is what `--json` writes and
what lets a report be re-rendered offline.

`METRICS_VERSION` (currently 2) must be bumped whenever the meaning of a measurement changes;
old cache entries are then ignored.

### trajectory.py: many pairs in order

`run_trajectory(sources, cache_dir)` runs the diff engine on each adjacent pair, in order, and
collects a `TrajectoryResult`. Because interval i needs checkpoints i and i+1, and checkpoint i
is still on disk from interval i−1, a two-file store downloads each checkpoint once.
`select_steps` picks the default ~25 steps: evenly spaced in log(step + 1), snapped to steps
that exist, always including the first and last.

### report.py and trajectory_report.py: HTML

Both render one self-contained HTML string: inline CSS and SVG, no JavaScript, no external
requests. Tooltips are SVG `<title>` elements. All text from a checkpoint (tensor names, paths)
is HTML-escaped. Light, dark and print styles are included.

- Pair report: banner, a heatmap in two panels (matrices, vectors) with their own log-spaced
  blue colour scales, "what moved most" tables, the noise floor section, and the label
  explanations.
- Trajectory report: banner, the short "reference scale, not a null" note, a two-panel
  heatmap with one row per (layer, component, kind) and one column per interval (each panel's
  scale is fixed across all intervals, so columns are comparable), a line chart per component,
  an interval table, the noise floor, and interval lengths.

Cells at or below the floor or the control are grey and hatched. Cells that moved away from
exactly zero have no relative change and are drawn as dashed outlines marked "0→".

## Getting the files

### sources.py: what an argument means

`resolve(arg)` turns a CLI argument into a source whose sha256 is known before any download:

1. An existing local path wins (a file, or a directory with `model.safetensors`).
2. A Windows path (`D:\...`) can't exist inside the container, so it is an error that points
   at the `/data` mount.
3. `org/name@revision` becomes a `HubSource`: only metadata is fetched now.
4. Anything else is reported as a missing path or an unknown format.

### hub.py: Hugging Face with the standard library

`fetch_metadata` reads a revision's file list and takes `model.safetensors`' size and LFS
sha256. `list_branches` lists the `stepN` branches for `trajectory org/name`. `download(remote,
dest)` streams to `dest.part`, resumes with an HTTP Range request if the `.part` exists, hashes
while streaming, and renames to `dest` only if the sha256 matches. On a mismatch the `.part` is
deleted.

`HF_TOKEN` is read from the environment only, sent only to the Hub host, dropped when the Hub
redirects to the download host, and never logged. `HF_ENDPOINT` overrides the Hub URL, which is
how the tests use a local fake Hub.

### store.py: at most N files on disk

`CheckpointStore` keeps downloads under `checkpoints/<org>/<name>/<revision>/`, with a
`checkpoint.json` holding the verified sha256, size and last-use time. `store.use(remote)` is a
context manager:

- If the stored file matches the Hub's sha256 and size, it is used as is.
- Otherwise the store first checks that it will be able to make room. If every stored file is
  in use, that is an error. It then downloads into the `.part`, and only after the new file is
  verified and renamed does it evict the least recently used file that is not in use.
- Files in use (inside a `with store.use(...)` block) are pinned and never evicted.

So a failed download never removes a stored file. The price is that during a download the
store holds up to N complete files plus one `.part`, and N + 1 complete files for a moment
after the rename. At rest it is back to N (default 2).

### cli.py: the plan and the guard

Before anything is downloaded, `planned_downloads` works out which files the command will
really fetch, in order. Pairs already in the metrics cache need no files, byte-identical pairs
need only A, and the store's pinning and eviction are replayed, so a file that is stored now
but evicted before its turn is counted twice. `download_guard` prints the count and total
size. Over 2 GB it needs confirmation: on a terminal it downloads the first file to measure
throughput, prints an estimate, and asks; without a terminal it stops before downloading
anything unless `--yes` was given.

`main(argv)` maps errors to exit codes (see `errors.py`): 0 ok, 1 internal, 2 usage,
3 architecture mismatch, 4 input or download error.

## Where to look when changing things

| You want to... | Start in |
|---|---|
| Support another architecture's tensor names | `arch.py` (`_RULES`), plus a test in `tests/unit/test_arch.py` |
| Change what a metric means | `metrics.py`, then bump `METRICS_VERSION` |
| Change the floor or a label threshold | `noise.py`, and update [metrics.md](metrics.md) |
| Change how a report looks | `report.py` / `trajectory_report.py` |
| Change download or eviction behaviour | `hub.py`, `store.py`, and `planned_downloads` in `cli.py` together |

Tests use tiny synthetic checkpoints written by a small helper in `tests/conftest.py` (the
reference `safetensors` library checks the reader in `tests/unit/test_reader.py`), and a
local fake Hub (`tests/fakehub.py`) that can serve, corrupt or cut off a file. No test touches
the real network unless it is marked `network`.
