# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project skeleton: src layout, PEP 621 `pyproject.toml`, Apache-2.0 license.
- `trajectory-explorer --version`.
- Docker image (`runtime` and `dev` stages) and compose services `te`, `test`, `lint`,
  `format`. Generated data goes to the required `TE_DATA_DIR` bind mount.
- GitHub Actions CI: ruff lint/format check and pytest, both run through docker compose.
