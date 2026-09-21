"""Command-line interface.

Only ``--version`` and ``--help`` exist so far; subcommands arrive in later chunks.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from trajectory_explorer._version import __version__

PROG = "trajectory-explorer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Weight-only diff tool for model checkpoints. Compares two checkpoints of the "
            "same architecture, or a training trajectory, tensor by tensor."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
