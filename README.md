# trajectory-explorer

**Weight-only diffs between model checkpoints: see *when* each part of a model learns.**

> Status: pre-alpha, under active development. Only `--version` works so far.

trajectory-explorer compares checkpoints of the same architecture tensor by tensor, without
running the model. Its headline use case is the [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m)
training trajectory: EleutherAI publishes 154 intermediate checkpoints as Hugging Face
revisions (`step0` … `step143000`). It also diffs any two same-architecture checkpoints,
e.g. `pythia-70m` vs `pythia-70m-deduped`, or `SmolLM2-135M` vs `SmolLM2-135M-Instruct`.

## Planned features

- Per-tensor relative delta norm `‖ΔW‖ / ‖W‖` and effective rank of each delta, grouped by
  layer and component (attention QKV, attention output, MLP in/out, embeddings, layer norms).
- Layer × component heatmap for a pair of checkpoints, plus a trajectory view across steps.
- A ranked "what moved most" table with labels: low-rank vs dense, concentrated vs spread out.
- An honest noise floor: effects below it are visually muted, and identical inputs report
  "no significant difference".
- Output: one self-contained, offline HTML file with inline SVG. No JavaScript framework.
- Low resource use: tensors stream one at a time, and at most two checkpoints are on disk.

## Quickstart (Docker)

You only need [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows).
No local Python needed. These commands work in PowerShell and bash, from the repo root.

1. Pick a folder for generated data (checkpoints, caches, reports) on a drive with a few GB
   free, create it, and point `.env` at it:

   ```powershell
   New-Item -ItemType Directory -Force D:\trajectory-explorer-data
   Copy-Item .env.example .env      # then edit TE_DATA_DIR in .env if needed
   ```

   `TE_DATA_DIR` is required. Without it, compose stops with a clear error, so nothing is
   written to an unexpected place.

2. Build and run:

   ```powershell
   docker compose build
   docker compose run --rm te --version
   docker compose run --rm te --help
   ```

## Development

| Task | Command |
|---|---|
| Run tests | `docker compose run --rm test` |
| Include real-model tests (downloads) | `docker compose run --rm test -m network` |
| Lint + format check | `docker compose run --rm lint` |
| Auto-fix and format | `docker compose run --rm format` |

The `test`, `lint` and `format` services mount your working tree, so you don't need to rebuild
after code changes. Rebuild with `docker compose build` after changing `pyproject.toml` or the
`Dockerfile`.

### Cleaning up

Remove this project's images and network, leaving other Docker data alone:

```powershell
docker compose down --rmi all
```

Every image is also labelled `io.trajectory-explorer.project=true`, so
`docker image ls --filter "label=io.trajectory-explorer.project=true"` lists exactly this
project's images. Generated data lives only in `TE_DATA_DIR`. Delete that folder to remove it.

## License

[Apache-2.0](LICENSE)
