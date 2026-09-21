"""Command-line interface.

Exit codes: 0 ok, 1 internal error, 2 bad arguments, 3 architecture mismatch,
4 missing or unreadable input.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from trajectory_explorer import hub
from trajectory_explorer._version import __version__
from trajectory_explorer.diff import (
    CheckpointSource,
    DiffOptions,
    DiffResult,
    diff_checkpoints,
    files_needed,
)
from trajectory_explorer.errors import InputError, TrajectoryExplorerError
from trajectory_explorer.report import render_html, summary_sentence
from trajectory_explorer.sources import HubSource, resolve
from trajectory_explorer.store import DEFAULT_MAX_CHECKPOINTS, CheckpointStore
from trajectory_explorer.trajectory import (
    TrajectoryResult,
    run_trajectory,
    select_steps,
    step_branches,
)
from trajectory_explorer.trajectory_report import render_trajectory_html
from trajectory_explorer.trajectory_report import summary_sentence as trajectory_summary

PROG = "trajectory-explorer"
log = logging.getLogger("trajectory_explorer")

# Downloads above this total need confirmation (decision D13): a y/N prompt or --yes.
GUARD_BYTES = 2 * 10**9


class UsageError(TrajectoryExplorerError):
    exit_code = 2


# -- argument helpers ---------------------------------------------------------------------
def _max_checkpoints(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("a diff needs at least 2 checkpoints on disk at once")
    return number


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


# -- download guard -----------------------------------------------------------------------
def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def planned_downloads(
    pairs: Sequence[tuple[CheckpointSource, CheckpointSource]], cache_dir: Path | None
) -> list[HubSource]:
    """Hub files these pairs will actually download, in the order the engine needs them.

    Pairs that are metrics-cache hits need no files; byte-identical pairs need only A.
    """
    needed = [source for a, b in pairs for source in files_needed(a, b, cache_dir)]
    pending: list[HubSource] = []
    for source in needed:
        new = all(p.info.path != source.info.path for p in pending)
        if isinstance(source, HubSource) and source.needs_download() and new:
            pending.append(source)
    return pending


def download_guard(pending: list[HubSource], *, yes: bool) -> None:
    """Announce planned downloads; over GUARD_BYTES, measure, estimate and ask first."""
    if not pending:
        return
    total = sum(s.info.size_bytes for s in pending)
    print(f"Downloads needed: {len(pending)} file(s), {_gb(total)} in total.", file=sys.stderr)
    if total <= GUARD_BYTES:
        return
    if not yes and not _stdin_is_tty():
        raise InputError(
            f"That is more than the {_gb(GUARD_BYTES)} allowed without confirmation, and there "
            "is no terminal to ask. Nothing was downloaded. Re-run with --yes to allow it."
        )
    first = pending[0]
    start = time.perf_counter()
    with first.fetch():
        pass
    rate = first.info.size_bytes / max(time.perf_counter() - start, 1e-6)
    rest = total - first.info.size_bytes
    print(
        f"Measured {rate / 1e6:.1f} MB/s on the first file. The remaining {len(pending) - 1} "
        f"file(s), {_gb(rest)}, should take about {_duration(rest / rate)}.",
        file=sys.stderr,
    )
    if yes:
        return
    answer = input("Continue downloading? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        raise InputError("Download not confirmed. Nothing more was downloaded.")


# -- commands -----------------------------------------------------------------------------
def _cache_dir() -> Path | None:
    return _data_dir() / "metrics" if "TE_DATA_DIR" in os.environ else None


def _store(args: argparse.Namespace) -> CheckpointStore:
    return CheckpointStore(
        _data_dir() / "checkpoints",
        args.max_checkpoints,
        announce=lambda message: print(message, file=sys.stderr),
    )


def _write_outputs(args: argparse.Namespace, html: str, json_text: str, default: Path) -> None:
    output = Path(args.output) if args.output else default
    _write(output, html)
    print(f"Report: {output}{_host_hint(output)}")
    if args.json:
        json_path = Path(args.json)
        _write(json_path, json_text)
        print(f"JSON:   {json_path}{_host_hint(json_path)}")


def cmd_diff(args: argparse.Namespace) -> int:
    store = _store(args)
    a, b = resolve(args.a, store), resolve(args.b, store)
    control = None
    if args.control:
        control = (resolve(args.control[0], store), resolve(args.control[1], store))
    cache_dir = _cache_dir()

    pairs = [(a, b)]
    if (
        control
        and a.info.sha256 != b.info.sha256
        and control[0].info.sha256 != control[1].info.sha256
    ):
        pairs.append(control)  # the engine skips the control for an identical main pair
    download_guard(planned_downloads(pairs, cache_dir), yes=args.yes)
    result = diff_checkpoints(a, b, DiffOptions(cache_dir=cache_dir, control=control))

    _write_outputs(args, render_html(result), result.to_json(), default_report_path(result))
    print(summary_sentence(result)[1])
    return 0


_REPO_ONLY = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def parse_steps(value: str) -> str | list[int]:
    """'default', 'all', or a comma-separated list of step numbers."""
    if value in ("default", "all"):
        return value
    try:
        steps = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected default, all or a list like 1000,2000,4000; got {value!r}"
        ) from None
    if len(steps) < 2:
        raise argparse.ArgumentTypeError("a step list needs at least two steps")
    return steps


def _hub_trajectory_specs(repo: str, steps_arg: str | list[int]) -> list[str]:
    available = step_branches(hub.list_branches(repo))
    if not available:
        raise InputError(
            f"{repo} has no step branches (named stepN). Pass an explicit ordered list of "
            f"checkpoints instead, e.g. trajectory-explorer trajectory {repo}@rev1 {repo}@rev2 ..."
        )
    if steps_arg == "all":
        chosen = available
    elif steps_arg == "default":
        chosen = select_steps(available)
    else:
        missing = [s for s in steps_arg if s not in set(available)]
        if missing:
            raise InputError(
                f"{repo} has no branch for step(s) {', '.join(map(str, missing))}. Available: "
                f"{available[0]} ... {available[-1]} ({len(available)} step branches)."
            )
        chosen = sorted(set(steps_arg))
    print(
        f"Steps ({len(chosen)} of {len(available)} step branches): " + ", ".join(map(str, chosen)),
        file=sys.stderr,
    )
    return [f"{repo}@step{s}" for s in chosen]


def default_trajectory_path(result: TrajectoryResult) -> Path:
    first, last = result.points[0], result.points[-1]
    digest = hashlib.sha256("".join(p.sha256 for p in result.points).encode()).hexdigest()[:8]
    name = (
        f"trajectory_{_safe(first.label)}_to_{_safe(last.label)}_"
        f"{len(result.points)}pts_{digest}.html"
    )
    return _data_dir() / "reports" / name


def cmd_trajectory(args: argparse.Namespace) -> int:
    specs: list[str] = args.specs
    if len(specs) == 1:
        (spec,) = specs
        if Path(spec).exists() or not _REPO_ONLY.match(spec):
            raise UsageError(
                "A trajectory needs either org/name (to use the repo's step branches) or two "
                f"or more checkpoints in order; got only {spec!r}."
            )
        specs = _hub_trajectory_specs(spec, args.steps or "default")
    elif args.steps is not None:
        raise UsageError("--steps only applies to the org/name form (a single repo argument).")

    store = _store(args)
    sources = [resolve(spec, store) for spec in specs]
    cache_dir = _cache_dir()
    pairs = list(itertools.pairwise(sources))
    download_guard(planned_downloads(pairs, cache_dir), yes=args.yes)
    result = run_trajectory(sources, cache_dir)

    html = render_trajectory_html(result)
    _write_outputs(args, html, result.to_json(), default_trajectory_path(result))
    print(trajectory_summary(result)[1])
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
            "A). Each argument is a .safetensors file, a directory containing "
            "model.safetensors, or a Hugging Face source org/name@revision (e.g. "
            "EleutherAI/pythia-70m@step1000), which is downloaded to $TE_DATA_DIR/checkpoints. "
            "Buffers such as attention masks are skipped by name."
        ),
    )
    diff.add_argument("a", metavar="A", help="reference checkpoint (path or org/name@revision)")
    diff.add_argument("b", metavar="B", help="checkpoint to compare against A")
    diff.add_argument(
        "--control",
        metavar="A2:B2",
        type=parse_control,
        help="optional control pair used as a reference scale; changes at or below the "
        "control's relative change are muted. Not a null: e.g. adjacent training "
        "checkpoints really differ.",
    )
    _add_common_options(diff, "diff_<A>_vs_<B>_<hashes>.html")
    diff.set_defaults(func=cmd_diff)

    traj = sub.add_parser(
        "trajectory",
        help="diff each adjacent pair of an ordered list of checkpoints",
        description=(
            "Diff every adjacent pair (step i vs step i+1) of an ordered list of checkpoints "
            "and write one report. Either pass a single repo, org/name, to use its step "
            "branches (stepN; --steps picks which), or pass two or more checkpoints in order "
            "(paths or org/name@revision). Adjacent-step diffs are a reference scale, not a "
            "null: the model really learns between neighbouring checkpoints."
        ),
    )
    traj.add_argument(
        "specs", nargs="+", metavar="SPEC", help="org/name, or checkpoints in training order"
    )
    traj.add_argument(
        "--steps",
        type=parse_steps,
        metavar="default|all|N,N,...",
        help="with org/name: about 25 log-spaced step branches (default), every step branch "
        "(all), or an explicit list such as 1000,2000,4000",
    )
    _add_common_options(traj, "trajectory_<first>_to_<last>_<n>pts_<hash>.html")
    traj.set_defaults(func=cmd_trajectory)
    return parser


def _add_common_options(parser: argparse.ArgumentParser, default_name: str) -> None:
    parser.add_argument(
        "-o",
        "--output",
        metavar="HTML",
        help=f"report path (default: $TE_DATA_DIR/reports/{default_name})",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the raw result as JSON")
    parser.add_argument(
        "--yes",
        action="store_true",
        help=f"allow downloads over {GUARD_BYTES // 10**9} GB without asking",
    )
    parser.add_argument(
        "--max-checkpoints",
        metavar="N",
        type=_max_checkpoints,
        default=DEFAULT_MAX_CHECKPOINTS,
        help="downloaded checkpoints kept on disk at once, least recently used evicted first "
        f"(default: {DEFAULT_MAX_CHECKPOINTS})",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug"
    )


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
