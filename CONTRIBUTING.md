# Contributing

Thanks for your interest. The project is early (0.1.0), so expect things to move.

## Ground rules

- **Everything runs in Docker.** You need Docker (Docker Desktop on Windows or macOS) and
  nothing else: no local Python, make or virtualenv. Every command below works in PowerShell
  and the same way in bash.
- **Keep dependencies minimal.** The runtime image depends on numpy only. The safetensors
  format and the Hub are handled by our own code (`reader.py`, `hub.py`). `safetensors`,
  `pytest` and `ruff` are dev-only. A new dependency needs a strong reason, stated in the pull
  request.
- **Keep memory and disk use low.** Load one tensor at a time as float32, never two full
  models. Downloaded checkpoints go through the rolling store (at most two on disk by
  default). Cache derived numbers, not weights.
- **Generated data goes under `TE_DATA_DIR`** (mounted at `/data`), never into the working
  tree.

## Setup

```powershell
Copy-Item .env.example .env        # set TE_DATA_DIR to an existing folder
docker compose build               # builds the runtime image and the dev image
```

The `test`, `lint` and `format` services mount your working tree at `/app`, so code changes
don't need a rebuild. Rebuild after changing `pyproject.toml` or the `Dockerfile`.

## Tests and lint

There is no CI. Run these locally before every commit and pull request:

```powershell
docker compose run --rm test        # the default suite; no network, a few seconds
docker compose run --rm lint        # ruff check + ruff format --check
docker compose run --rm format      # auto-fix: ruff format, then ruff check --fix
```

Arguments after `test` go to pytest, e.g. `docker compose run --rm test -k store -x`.

Tests are marked:

| Marker | Meaning | Runs by default |
|---|---|---|
| `unit` | one function, no files beyond tmp | yes |
| `integration` | several modules, tiny synthetic checkpoints, a local fake Hub | yes |
| `slow` | longer runs | no |
| `network` | downloads real checkpoints from the Hugging Face Hub | no |

`docker compose run --rm test -m network` runs the opt-in real-Pythia test. It downloads real
files into `TE_DATA_DIR`.

When adding a test:

- New behaviour gets a test. Keep the suite small and meaningful.
- Use the helpers in `tests/conftest.py` for synthetic checkpoints, and `tests/fakehub.py`
  for anything that talks to the Hub. The fake Hub can serve a file, corrupt it, or drop the
  connection partway. The default suite blocks the real Hub, so a test can't download by
  accident.
- Never put real weights, reports or other generated data in the repository.

## Changing metrics

If you change what a measurement means, bump `METRICS_VERSION` in `metrics.py`. That makes old
cache entries miss instead of silently mixing old and new numbers. If you change the floor or
a label threshold in `noise.py`, update [docs/metrics.md](docs/metrics.md) and say in the pull
request what it changes on a real checkpoint pair.

## Commits

Plain, short commit messages, one line, in the imperative mood, with no type prefix (no
`feat:` or `fix:`) and no emoji. For example:

    Add tensor reader for float32, float16 and bfloat16

Add a body only when the reason for a change isn't obvious. Keep commits small and focused.
Don't rewrite history that has already been pushed.

## Pull requests

Before opening one:

1. `docker compose run --rm test` passes.
2. `docker compose run --rm lint` passes.
3. New behaviour has a test, and user-visible changes are noted in `CHANGELOG.md` under
   "Unreleased".
