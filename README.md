# trajectory-explorer

**Weight-only diffs between model checkpoints: see where, and when, a model changed.**

> Status: 0.1.0, early. Tested end to end on Pythia-70M only. Docker-only; there is no PyPI
> package.

trajectory-explorer compares two checkpoints of the same architecture tensor by tensor, from
the weights alone. It never runs the model. For each tensor it measures how much it changed
relative to where it started, how many directions that change really uses, and whether the
change is bigger than what storage rounding alone could produce. Then it writes one
self-contained HTML report.

## Why

When a model gets better at something during training, or changes after fine-tuning, the
change is somewhere in its weights. This tool is meant to trace it there: which layers and
which components (attention, MLP, embeddings, layer norms) moved, by how much, and between
which two checkpoints. The headline use case is the
[Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) training run. EleutherAI publishes
154 intermediate checkpoints as Hugging Face branches (`step0`, `step1`, … `step143000`), so
you can ask "when does each part of the model learn?" by diffing neighbouring checkpoints
along the run.

It is deliberately honest about what it can't tell you. A change that is smaller than rounding
is reported as such. Two identical files report "No significant difference", not a wall of
colour. A diff between neighbouring training steps is labelled a reference scale, not a
"nothing happened" baseline (see [below](#adjacent-step-diffs-are-a-reference-scale-not-a-null)).

## What you need

- [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows). No local Python,
  make or anything else.
- A folder for generated data (checkpoints, metrics cache, reports) on a drive with a few GB
  free. At most two checkpoints are kept on disk at once, plus one partial download while a
  new file is on its way. For Pythia-70M that is about 0.85 GB at the peak.
- About 0.5 GB of free RAM. The tool loads one tensor at a time as float32. Its peak memory
  on Pythia-70M was 433 MB (measured).

## Quickstart (PowerShell, fresh clone)

```powershell
git clone https://github.com/Aryan-Pillai7/trajectory-explorer.git
cd trajectory-explorer

# 1. Point TE_DATA_DIR at an existing folder. Everything the tool writes goes there.
New-Item -ItemType Directory -Force D:\trajectory-explorer-data
Copy-Item .env.example .env      # edit TE_DATA_DIR in .env if you want another folder

# 2. Build the image (python:3.12-slim + numpy).
docker compose build

# 3. Compare two neighbouring Pythia-70M checkpoints (downloads 2 x 281.7 MB).
docker compose run --rm te diff EleutherAI/pythia-70m@step142000 EleutherAI/pythia-70m@step143000

# 4. A short trajectory over three checkpoints (downloads 3 x 281.7 MB).
docker compose run --rm te trajectory EleutherAI/pythia-70m --steps 1000,2000,4000
```

Each command prints where it wrote the report, both as the path inside the container and as
the path on your machine, for example
`Report: /data/reports/diff_... (on the host: D:/trajectory-explorer-data/reports/diff_...)`.
Open that file in any browser. It works offline and has no JavaScript.

`TE_DATA_DIR` is required on purpose. Without it, compose stops with an error instead of
writing gigabytes to some default location.

The default trajectory, `docker compose run --rm te trajectory EleutherAI/pythia-70m`, picks
about 25 log-spaced steps and downloads about 7 GB in total. On this machine that was 25 files
at 7 to 11 MB/s. Because it is over 2 GB, the tool first downloads one file to measure your
throughput, prints an estimate and asks before going on (see [Downloads](#downloads-and-the-2-gb-guard)).

## Commands

Checkpoints can be given as:

- a Hugging Face revision, `org/name@revision` (e.g. `EleutherAI/pythia-70m@step1000`),
  downloaded into `TE_DATA_DIR/checkpoints`;
- a local `.safetensors` file, or a directory containing `model.safetensors`. The tool runs in
  a container, so local files must be under `TE_DATA_DIR` and are passed by their container
  path, e.g. `/data/my-models/run1/model.safetensors`. A Windows path like `D:\...` gets an
  error that says so.

Only `model.safetensors` is read. Checkpoints that only ship `pytorch_model.bin` are refused
with a clear error. Known buffers (attention masks, rotary tables) are skipped by name. Names
and shapes of all other tensors must match, or the tool stops with exit code 3 and lists the
differences.

### `diff A B`

Compares B against A. Relative changes are measured against A. Writes one report with:

- a banner sentence stating the largest matrix change and the largest vector change, and how
  many tensors are above the noise floor;
- a layer × component heatmap, in two panels (weight matrices; vectors, i.e. biases and norm
  scales), each with its own colour scale;
- "What moved most" tables, with effective rank and labels;
- the noise-floor section, and a section explaining the labels.

| Flag | What it does |
|---|---|
| `--control A2:B2` | A second pair used as a reference scale. Tensors and cells whose change is at or below the control pair's change are muted and reported as "below control". |
| `-o, --output HTML` | Report path. Default: `$TE_DATA_DIR/reports/diff_<A>_vs_<B>_<hashes>.html`. |
| `--json PATH` | Also write the full result as JSON. |
| `--yes` | Allow downloads over 2 GB without asking. |
| `--max-checkpoints N` | Downloaded checkpoints kept on disk at once, least recently used evicted first. Default 2. |
| `-v`, `-vv` | Progress, or debug output, on stderr. |

### `trajectory SPEC [SPEC ...]`

Diffs every adjacent pair (step i against step i+1) of an ordered list of checkpoints and
writes one report: a heatmap of (layer, component) rows × intervals, a line chart of relative
change per interval by component, a per-interval table, the noise floor, and a note on
interval lengths (intervals early in a run can be 1 step long, later ones tens of thousands).

Either pass one repo, `org/name`, to use its `stepN` branches, or pass two or more checkpoints
in training order.

| Flag | What it does |
|---|---|
| `--steps default\|all\|N,N,...` | With `org/name`: about 25 steps spaced evenly on a log scale (default; dense near the start, where the model changes fastest), every step branch (`all`), or an explicit list. |
| `-o, --output HTML` | Report path. Default: `$TE_DATA_DIR/reports/trajectory_<first>_to_<last>_<n>pts_<hash>.html`. |
| `--json PATH` | Also write the full result as JSON. |
| `--yes` | Allow downloads over 2 GB without asking. |
| `--max-checkpoints N` | As for `diff`. |
| `-v`, `-vv` | Progress, or debug output. |

`trajectory` has no `--control` flag.

Checkpoints are fetched in order, so with the two-file store each one is downloaded once. The
one exception: a checkpoint already in the store from an earlier run can be evicted before
its turn and fetched again; the download plan counts that.
Every interval goes through the same diff engine and metrics cache as `diff`. A rerun of a
finished trajectory downloads and measures nothing. It still makes a few small metadata
requests to the Hub, so it needs the network.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Internal error (a bug; please report it) |
| 2 | Bad arguments |
| 3 | The checkpoints do not have the same architecture (names or shapes differ) |
| 4 | Missing or unreadable input: file not found, not safetensors, Hub or network error, failed download or checksum, download not confirmed |

## Where data lives

Everything goes under `TE_DATA_DIR` on the host, mounted at `/data` in the container. On the
author's machine that is `D:\trajectory-explorer-data`.

| Folder | Contents |
|---|---|
| `checkpoints/<org>/<name>/<revision>/` | Downloaded `model.safetensors` plus a small `checkpoint.json` (sha256, size, last use). At most `--max-checkpoints` of them. |
| `metrics/` | Cached per-pair measurements as JSON, keyed by the two files' sha256 and the metrics version. Reports regenerate from these without downloading anything. |
| `reports/` | HTML reports (and JSON, if you ask for it with `--json`). |

Downloads resume from a `.part` file if interrupted, and every file is checked against the
Hub's sha256 before it is used. A stored checkpoint is evicted only after its replacement
download has succeeded, so a failed download never costs you a file. If a model is gated or
private, put `HF_TOKEN=...` in `.env`. The token is sent only to the Hub, never to the download
host it redirects to, and is never logged.

To delete everything the tool generated, delete that folder. To remove this project's images:
`docker compose down --rmi all`.

## Downloads and the 2 GB guard

Before downloading, both commands print how many files and bytes they will fetch. The plan
accounts for files that will be evicted and fetched again. If the total is over 2 GB:

- in an interactive terminal, the tool downloads the first file, measures the throughput,
  prints the estimated time for the rest and asks `Continue downloading? [y/N]`;
- with no terminal (for example piped or scripted), it stops with exit code 4 before
  downloading anything, unless you pass `--yes`.

## The noise floor, briefly

Two copies of the same weights can differ slightly just from being stored at different
precisions. The noise floor is the largest relative change that rounding alone can plausibly
explain: `3 × √(uA² + uB²) / √3`, where `u` is the unit roundoff of each side (float32 `2⁻²⁴`,
float16 `2⁻¹¹`, bfloat16 `2⁻⁸`). Changes at or below it are drawn grey and hatched and never
count as significant.

One refinement matters for Pythia. Its step checkpoints are stored as float32, but every value
in them is exactly a float16 number, so they really carry float16 precision. When both sides
of a tensor hold only float16 values, the float16 floor is used (0.12% instead of
0.000015%). Details, and the other metrics, are in [docs/metrics.md](docs/metrics.md).

## Adjacent-step diffs are a reference scale, not a null

It is tempting to treat the diff between two neighbouring training checkpoints as "noise" and
call everything bigger than it real. That is wrong: the model really learns between
`step142000` and `step143000`, so that diff is itself a real change. In the real run, 71 of 76
Pythia-70M tensors moved more than the noise floor over those 1,000 steps. It is useful as a
yardstick (`diff --control A2:B2` mutes whatever is smaller than it), but it is not a null.
The true nulls this tool relies on are rounding (the floor above), byte-identical files, and
checkpoints verified to be the same weights. Pythia-70M's `main` is bit for bit the float16
rounding of `step143000`, and the tool reports that pair as "No significant difference".

## Scope and limits

- Weight-only. No activations, no outputs, no inputs are run through the model.
- One file, `model.safetensors`. Sharded checkpoints are not supported yet.
- Tensor-to-component rules exist for GPT-NeoX (Pythia) and Llama-style models (such as
  SmolLM2); anything else falls back to a generic guess or "other". Only Pythia-70M has been
  run end to end.
- The rank and concentration labels are heuristics with fixed thresholds, checked so far
  mainly on Pythia-70M ([docs/metrics.md](docs/metrics.md)).
- Fused attention QKV is one tensor in Pythia and is reported as one component, not split
  into Q, K and V.

## More

- [docs/architecture.md](docs/architecture.md): how the pieces fit together.
- [docs/metrics.md](docs/metrics.md): what each number means and how it is computed.
- [examples/](examples/): a real diff run and what its report looks like.
- [CONTRIBUTING.md](CONTRIBUTING.md): development workflow (Docker only, tests, lint).
- [CHANGELOG.md](CHANGELOG.md)

## License

[Apache-2.0](LICENSE)
