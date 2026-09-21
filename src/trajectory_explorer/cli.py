"""Command-line interface.

Exit codes: 0 ok, 1 internal error, 2 bad arguments, 3 architecture mismatch,
4 missing or unreadable input.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from trajectory_explorer._version import __version__
from trajectory_explorer.diff import DiffOptions, DiffResult, diff_checkpoints
from trajectory_explorer.errors import InputError, TrajectoryExplorerError
from trajectory_explorer.report import render_html, summary_sentence

PROG = "trajectory-explorer"
log = logging.getLogger("trajectory_explorer")

# "org/name@revision", e.g. EleutherAI/pythia-70m@step1000
_HUB_SPEC = re.compile(r"^[\w.-]+/[\w.-]+@[\w./-]+$")


class UsageError(TrajectoryExplorerError):
    exit_code = 2


# -- argument helpers ---------------------------------------------------------------------
def resolve_local(spec: str) -> Path:
    """A .safetensors file, or a directory containing model.safetensors."""
    path = Path(spec)
    if path.is_dir():
        candidate = path / "model.safetensors"
        if not candidate.is_file():
            raise InputError(f"Directory {spec} has no model.safetensors")
        return candidate
    if path.is_file():
        return path
    if _HUB_SPEC.match(spec):
        raise InputError(
            f"{spec!r} looks like a Hugging Face source (repo@revision). Hub sources are not "
            "built yet: pass a local .safetensors file or a directory containing "
            "model.safetensors."
        )
    raise InputError(f"Checkpoint not found: {spec}")


def parse_control(value: str) -> tuple[str, str]:
    """Split 'A2:B2'. With several ':' (unusual paths), pick the split where both exist."""
    if value.count(":") == 1:
        first, second = value.split(":")
        if first and second:
            return first, second
    else:
        for i, char in enumerate(value):
            if char == ":" and Path(value[:i]).exists() and Path(value[i + 1 :]).exists():
                return value[:i], value[i + 1 :]
    raise argparse.ArgumentTypeError(
        f"expected A2:B2 (two checkpoints separated by ':'), got {value!r}"
    )


def _data_dir() -> Path:
    return Path(os.environ.get("TE_DATA_DIR", "/data"))


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "checkpoint"


def default_report_path(result: DiffResult) -> Path:
    name = (
        f"diff_{_safe(result.a.label)}_vs_{_safe(result.b.label)}_"
        f"{result.a.sha256[:8]}_{result.b.sha256[:8]}.html"
    )
    return _data_dir() / "reports" / name


def _host_hint(path: Path) -> str:
    """Where a /data path lives on the host, if compose told us (TE_HOST_DATA_DIR)."""
    host = os.environ.get("TE_HOST_DATA_DIR")
    try:
        relative = path.resolve().relative_to(_data_dir().resolve())
    except ValueError:
        return ""
    return f" (on the host: {Path(host) / relative})" if host else ""


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise UsageError(f"Cannot write {path}: {exc.strerror or exc}") from exc


# -- commands -----------------------------------------------------------------------------
def cmd_diff(args: argparse.Namespace) -> int:
    a, b = resolve_local(args.a), resolve_local(args.b)
    control = None
    if args.control:
        control = (resolve_local(args.control[0]), resolve_local(args.control[1]))
    cache_dir = _data_dir() / "metrics" if "TE_DATA_DIR" in os.environ else None

    result = diff_checkpoints(a, b, DiffOptions(cache_dir=cache_dir, control=control))

    output = Path(args.output) if args.output else default_report_path(result)
    _write(output, render_html(result))
    print(f"Report: {output}{_host_hint(output)}")
    if args.json:
        json_path = Path(args.json)
        _write(json_path, result.to_json())
        print(f"JSON:   {json_path}{_host_hint(json_path)}")
    print(summary_sentence(result)[1])
    return 0


# -- parser and entry point ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Weight-only diff tool for model checkpoints. Compares two checkpoints of the "
            "same architecture, tensor by tensor, and writes a self-contained HTML report."
        ),
        epilog="Exit codes: 0 ok, 1 internal error, 2 bad arguments, "
        "3 architecture mismatch, 4 missing or unreadable input.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    diff = sub.add_parser(
        "diff",
        help="compare two checkpoints and write an HTML report",
        description=(
            "Compare checkpoint B against checkpoint A (relative changes are measured against "
            "A). Each argument is a .safetensors file or a directory containing "
            "model.safetensors. Buffers such as attention masks are skipped by name."
        ),
    )
    diff.add_argument("a", metavar="A", help="reference checkpoint")
    diff.add_argument("b", metavar="B", help="checkpoint to compare against A")
    diff.add_argument(
        "-o",
        "--output",
        metavar="HTML",
        help="report path (default: $TE_DATA_DIR/reports/diff_<A>_vs_<B>_<hashes>.html)",
    )
    diff.add_argument(
        "--control",
        metavar="A2:B2",
        type=parse_control,
        help="optional control pair used as a reference scale; changes at or below the "
        "control's relative change are muted. Not a null: e.g. adjacent training "
        "checkpoints really differ.",
    )
    diff.add_argument("--json", metavar="PATH", help="also write the raw result as JSON")
    diff.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug"
    )
    diff.set_defaults(func=cmd_diff)
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on bad arguments, 0 on --version/--help
    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        return 2
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except TrajectoryExplorerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        log.debug("unexpected error", exc_info=True)
        print(f"internal error: {exc!r} (run with -vv for a traceback)", file=sys.stderr)
        return 1
