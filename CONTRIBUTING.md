# Contributing

Thanks for your interest! The project is pre-alpha, so expect things to move.

## Ground rules

- **Everything runs in Docker.** You don't need Python, make or anything else on the host,
  only Docker (Docker Desktop on Windows/macOS). Commands are documented for PowerShell and
  work the same in bash.
- **Keep dependencies minimal.** Runtime dependencies are `numpy` and `safetensors`. A new
  dependency needs a strong reason, stated in the pull request.
- **Keep memory and disk use low.** Stream tensors one at a time; never load two full models.

## Workflow

The build, run, test and lint commands are in the README. Before opening a pull request:

1. Tests pass: `docker compose run --rm test`
2. Lint and format pass: `docker compose run --rm lint`
   (fix formatting with `docker compose run --rm format`)
3. New behaviour has a test. Mark it `unit`, `integration`, `slow` or `network`.
   Tests that download real checkpoints must be marked `network`.

## Commits

Write short, plain commit messages in the imperative mood, with no type prefix. For example:
"Add tensor reader for float32, float16 and bfloat16". Add a body only when the reason for a
change isn't obvious. Keep commits small and focused.
