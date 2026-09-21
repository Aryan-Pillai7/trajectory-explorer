"""Exception hierarchy. Each error carries the process exit code the CLI should use.

Exit codes: 0 ok, 1 internal error, 2 usage error (argparse), 3 incompatible architectures,
4 input error (missing/unreadable/malformed checkpoint, download failure).
"""

from __future__ import annotations

from collections.abc import Sequence


class TrajectoryExplorerError(Exception):
    """Base class for errors that should be reported to the user without a traceback."""

    exit_code: int = 1


class InputError(TrajectoryExplorerError):
    """A checkpoint could not be found, read or downloaded."""

    exit_code = 4


class CheckpointFormatError(InputError):
    """A file is not a valid safetensors checkpoint."""


class UnsupportedDtypeError(CheckpointFormatError):
    """A tensor has a dtype this tool cannot convert to float32."""


class ArchitectureMismatch(TrajectoryExplorerError):
    """Two checkpoints do not share the same parameter names and shapes."""

    exit_code = 3

    def __init__(
        self,
        only_in_a: Sequence[str],
        only_in_b: Sequence[str],
        shape_mismatches: Sequence[tuple[str, tuple[int, ...], tuple[int, ...]]],
        *,
        max_listed: int = 10,
        header: str = "The two checkpoints do not have the same architecture.",
        labels: tuple[str, str] = ("A", "B"),
    ) -> None:
        self.header = header
        self.labels = labels
        self.only_in_a = list(only_in_a)
        self.only_in_b = list(only_in_b)
        self.shape_mismatches = list(shape_mismatches)
        super().__init__(self._format(max_listed))

    def _format(self, max_listed: int) -> str:
        lines = [self.header]
        la, lb = self.labels

        def listing(title: str, items: list[str]) -> None:
            if not items:
                return
            lines.append(f"{title} ({len(items)}):")
            lines.extend(f"  - {item}" for item in items[:max_listed])
            if len(items) > max_listed:
                lines.append(f"  ... and {len(items) - max_listed} more")

        listing(f"Only in {la}", self.only_in_a)
        listing(f"Only in {lb}", self.only_in_b)
        listing(
            "Shape differs",
            [f"{name}: {la}{list(sa)} vs {lb}{list(sb)}" for name, sa, sb in self.shape_mismatches],
        )
        return "\n".join(lines)
