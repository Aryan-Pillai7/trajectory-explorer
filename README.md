# trajectory-explorer

**See which parts of an AI model changed, and when, just by comparing its saved weights.**

> Status: 0.1.0, early. Tested end to end on Pythia-70M only. Runs in Docker. There is no
> PyPI package.

## What is this?

An AI model is a large collection of numbers called **weights**, stored in groups called
**tensors**. During training these numbers keep changing. Many projects save a copy of the
model along the way. Each copy is a **checkpoint**.

trajectory-explorer takes two checkpoints of the same model and compares them number by
number. It never runs the model. It only looks at the saved weights. It then tells you:

- **where** the model changed: which layer, and which part (attention, MLP, embeddings,
  layer norms);
- **how much** each part changed, compared with its size at the start;
- **what kind** of change it was: spread across the whole part, or packed into a few
  directions or a few rows;
- **whether the change is real**, or so small that it could just be rounding in how the numbers
  are stored.

It writes the answer as one HTML page with colour maps and tables. You open it in any browser,
and it works offline.

## Why?

When a model learns something, the change is somewhere in its weights. This tool helps you
trace it there.

The main example is **Pythia-70M**, a small language model. Its makers (EleutherAI) published
154 checkpoints from its training run, from `step0` to `step143000`. Compare neighbouring
checkpoints along the run and you can ask: *when does each part of the model learn?*

The tool also compares any two models with the same structure, such as a base model and its
fine-tuned version.

## What you get

`diff` compares two checkpoints. Its report has:

- a one-sentence summary at the top, for example: *"The largest weight-matrix change is in
  unembed (86.8%); the largest vector change is in layer norm (83.3%). 76 of 76 tensors changed
  more than the noise floor."*;
- a colour map, with one row per layer and one column per model part. Darker means more change;
- a table of what moved most;
- short explanations of every number.

`trajectory` compares a whole series of checkpoints, one neighbouring pair at a time, and puts
them on one page. Each column of its colour map is one stretch of training, so you can watch
different parts of the model start and stop changing.

A real example with a screenshot is in [examples/](examples/).

## Being honest about small changes

Two ideas keep the report from exaggerating.

**1. The noise floor.** Computers store numbers with limited precision, so a tiny difference
can come from rounding alone. The tool works out how big that rounding could be and calls
anything at or below it "not significant". Those cells are drawn grey. Compare a file with
itself and you get "No significant difference", not a page full of colour.

Some files are saved in a precise format (float32), but the numbers inside only carry a less
precise format's worth of detail (float16). Pythia's checkpoints are like this. The tool checks
for it and uses the matching, larger rounding limit, so it doesn't mistake rounding for
learning.

**2. Neighbouring checkpoints are a yardstick, not "zero".** It is tempting to say the change
between two neighbouring checkpoints is just noise. It isn't: the model really learns between
them. For example, between step 142000 and step 143000 of Pythia-70M, 71 of 76 tensors changed
more than rounding can explain. So you can use such a pair as a **reference scale**, and hide
anything smaller with `--control`. It doesn't show that nothing happened.

## Good to know

- It compares weights only. It doesn't run the model or look at its answers.
- It reads `model.safetensors` files only, one file per checkpoint. Models that only ship
  `pytorch_model.bin` are refused with a clear message.
- It knows the part names of Pythia (GPT-NeoX) and Llama-style models such as SmolLM2. Others
  fall back to a best guess. Only Pythia-70M has been tested end to end.
- The labels (low-rank/dense, concentrated/spread) use fixed rule-of-thumb thresholds. So far
  they have only been checked on Pythia-70M. Treat them as hints and look at the numbers next
  to them.
- It is light on your computer. It loads one tensor at a time (peak memory on Pythia-70M was
  433 MB) and keeps at most two downloaded checkpoints on disk.

---

## Try it yourself

These steps are for Windows PowerShell. They work the same way in a Mac or Linux terminal,
except step 3.

### 1. Install Docker

Install [Docker Desktop](https://docs.docker.com/get-docker/) and start it. You don't need
Python or anything else. Everything runs inside Docker.

### 2. Get the code

```powershell
git clone https://github.com/Aryan-Pillai7/trajectory-explorer.git
cd trajectory-explorer
```

### 3. Choose a folder for the tool's files

The tool saves downloaded checkpoints, cached results and reports in one folder. Pick a drive
with a few GB free. This example uses `D:`.

```powershell
New-Item -ItemType Directory -Force D:\trajectory-explorer-data
Copy-Item .env.example .env
```

Open `.env` and check that `TE_DATA_DIR` points at your folder, for example
`TE_DATA_DIR=D:/trajectory-explorer-data`. This setting is required: without it the tool stops
instead of writing files somewhere unexpected.

### 4. Build

```powershell
docker compose build
```

This takes a few minutes the first time.

### 5. Compare two checkpoints

```powershell
docker compose run --rm te diff EleutherAI/pythia-70m@step142000 EleutherAI/pythia-70m@step143000
```

This downloads two Pythia-70M checkpoints (281.7 MB each) and compares them. At the end it
prints the summary and where the report went, for example:

```
Report: /data/reports/diff_... (on the host: D:/trajectory-explorer-data/reports/diff_...)
```

Open the "on the host" file in your browser. Run the same command again and it finishes in
seconds, because the results are cached.

### 6. Follow a short stretch of training

```powershell
docker compose run --rm te trajectory EleutherAI/pythia-70m --steps 1000,2000,4000
```

This compares step 1000 → 2000 and 2000 → 4000 and writes one report for both (3 downloads,
281.7 MB each).

### 7. Optional: the full training run

```powershell
docker compose run --rm te trajectory EleutherAI/pythia-70m
```

This picks about 25 checkpoints across the whole run and downloads about 7 GB in total. On the
author's machine that ran at 7 to 11 MB/s. Because it is over 2 GB, the tool first downloads one
file, estimates the total time and asks `Continue downloading? [y/N]` before going on. Only two
checkpoints are kept on disk at any time.

### 8. Optional: run the tests

```powershell
docker compose run --rm test
docker compose run --rm lint
```

The tests take a few seconds and use tiny made-up checkpoints. Nothing is downloaded.

## Using your own checkpoints

Put the files inside your data folder and use the path as the container sees it. `D:\trajectory-explorer-data`
becomes `/data`:

```powershell
docker compose run --rm te diff /data/my-models/run1 /data/my-models/run2
```

Each path can be a `.safetensors` file or a folder containing `model.safetensors`. Hugging Face
models are written as `org/name@revision`. For gated or private models, add `HF_TOKEN=...` to
`.env`. The token is only sent to Hugging Face and is never written to logs.

## Options

| Option | Works with | What it does |
|---|---|---|
| `--control A2:B2` | `diff` | Use a second pair as a yardstick: changes no bigger than that pair's are greyed out |
| `--steps default\|all\|N,N,...` | `trajectory` | About 25 steps (default), every step (`all`), or your own list |
| `-o PATH` | both | Where to write the report (default: `reports/` in your data folder) |
| `--json PATH` | both | Also save the full results as JSON |
| `--yes` | both | Allow downloads over 2 GB without asking |
| `--max-checkpoints N` | both | How many downloaded checkpoints to keep on disk (default 2) |
| `-v`, `-vv` | both | Show progress, or full debug output |

`docker compose run --rm te diff --help` and `... te trajectory --help` show the same list.

## If something goes wrong

The tool ends with an exit code that says what happened:

| Code | Meaning |
|---|---|
| 0 | Worked |
| 1 | A bug in the tool. Please report it |
| 2 | The command was typed wrong |
| 3 | The two checkpoints have a different structure, so they can't be compared |
| 4 | A file is missing or unreadable, a download failed, or a large download wasn't confirmed |

Interrupted downloads pick up where they stopped next time. Every download is checked against
Hugging Face's checksum before use. A stored checkpoint is only deleted after its replacement
has downloaded successfully.

## Where files go and how to clean up

Everything goes into your data folder:

- `checkpoints/`: downloaded models (at most two at a time);
- `metrics/`: cached results, so reports can be rebuilt without downloading again;
- `reports/`: the HTML reports.

To remove it all, delete the folder. To remove this project's Docker images:

```powershell
docker compose down --rmi all
```

## Learn more

- [docs/metrics.md](docs/metrics.md): what every number means and how it is calculated.
- [docs/architecture.md](docs/architecture.md): how the code is organised.
- [examples/](examples/): a real run and a screenshot of its report.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to work on the code.
- [CHANGELOG.md](CHANGELOG.md)

## License

[Apache-2.0](LICENSE)
