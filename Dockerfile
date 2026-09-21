# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# base: Python + runtime dependencies only (numpy, safetensors).
# Dependencies are read from pyproject.toml so it stays the single source of
# truth, and they form their own cached layer that only rebuilds when
# pyproject.toml changes.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

LABEL io.trajectory-explorer.project="true" \
      org.opencontainers.image.title="trajectory-explorer" \
      org.opencontainers.image.description="Weight-only diff tool for model checkpoints" \
      org.opencontainers.image.source="https://github.com/Aryan-Pillai7/trajectory-explorer" \
      org.opencontainers.image.licenses="Apache-2.0"

# No pip cache, no bytecode: nothing grows inside Docker's disk image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TE_DATA_DIR=/data \
    HF_HOME=/data/hf

WORKDIR /app

COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print('\n'.join(d['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && useradd --create-home --uid 1000 te

# ---------------------------------------------------------------------------
# runtime: the CLI image (`docker compose run --rm te ...`).
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && rm -rf /app/src /app/build

USER te
ENTRYPOINT ["trajectory-explorer"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# dev: adds pytest + ruff and an editable install. compose bind-mounts the repo
# at /app, so tests and lint always see the live source without rebuilding.
# ---------------------------------------------------------------------------
FROM base AS dev

RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print('\n'.join(d['project']['optional-dependencies']['dev']))" > /tmp/requirements-dev.txt \
 && pip install --no-cache-dir -r /tmp/requirements-dev.txt \
 && rm /tmp/requirements-dev.txt

COPY README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir --no-deps -e .

ENV RUFF_CACHE_DIR=/tmp/ruff-cache
USER te
