"""Rolling on-disk store of downloaded checkpoints: at most N complete files (default 2).

Layout: <root>/<org>/<name>/<revision>/model.safetensors plus a small checkpoint.json with
the verified sha256, size and last-use time. Only files that were verified and renamed into
place (hub.download) count; ``*.part`` files are resumed by the next download of the same
revision or removed when room is needed. A checkpoint in use is pinned and never evicted;
otherwise the least recently used one is evicted to make room for a new one.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from trajectory_explorer import hub
from trajectory_explorer.errors import InputError
from trajectory_explorer.hub import DownloadStats, HubSpec, RemoteFile

log = logging.getLogger(__name__)

META_NAME = "checkpoint.json"
DEFAULT_MAX_CHECKPOINTS = 2
Downloader = Callable[[RemoteFile, Path], DownloadStats]


class CheckpointStore:
    def __init__(
        self,
        root: Path,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
        *,
        downloader: Downloader | None = None,
        announce: Callable[[str], None] | None = None,
    ) -> None:
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1")
        self.root = root
        self.max_checkpoints = max_checkpoints
        self._download = downloader or (lambda remote, dest: hub.download(remote, dest))
        self._announce = announce or (lambda message: log.info("%s", message))
        self._pinned: set[Path] = set()

    # -- layout -----------------------------------------------------------------------
    def path_for(self, spec: HubSpec) -> Path:
        org, name = spec.repo.split("/", 1)
        return self.root / org / name / quote(spec.revision, safe="") / hub.FILENAME

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_name(META_NAME)

    def _read_meta(self, path: Path) -> dict | None:
        try:
            return json.loads(self._meta_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_meta(self, path: Path, remote: RemoteFile) -> None:
        meta = {
            "repo": remote.spec.repo,
            "revision": remote.spec.revision,
            "sha256": remote.sha256,
            "size": remote.size,
            "last_used_ns": time.time_ns(),
        }
        self._meta_path(path).write_text(json.dumps(meta), encoding="utf-8")

    def complete(self) -> list[Path]:
        """Verified checkpoints currently on disk (never .part files)."""
        if not self.root.is_dir():
            return []
        return sorted(
            p for p in self.root.rglob(hub.FILENAME) if p.is_file() and self._read_meta(p)
        )

    def lru_order(self) -> list[Path]:
        """Stored checkpoints, least recently used first (the order eviction uses)."""
        return sorted(
            self.complete(), key=lambda p: (self._read_meta(p) or {}).get("last_used_ns", 0)
        )

    def lookup(self, remote: RemoteFile) -> Path | None:
        """The stored file for this exact content, or None."""
        path = self.path_for(remote.spec)
        meta = self._read_meta(path) if path.is_file() else None
        if meta and meta.get("sha256") == remote.sha256 and meta.get("size") == remote.size:
            return path
        return None

    # -- eviction ---------------------------------------------------------------------
    def _remove(self, path: Path) -> None:
        for p in (path, self._meta_path(path), path.with_name(path.name + ".part")):
            p.unlink(missing_ok=True)
        folder = path.parent
        while folder != self.root and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent

    def _make_room(self, incoming: Path) -> None:
        # Stale partial downloads of other revisions: remove (only `incoming` gets resumed).
        if self.root.is_dir():
            for part in self.root.rglob(hub.FILENAME + ".part"):
                if part.with_name(hub.FILENAME) != incoming:
                    log.info("Removing stale partial download %s", part)
                    part.unlink(missing_ok=True)
        others = [p for p in self.complete() if p != incoming]
        while len(others) >= self.max_checkpoints:
            unpinned = [p for p in others if p not in self._pinned]
            if not unpinned:
                raise InputError(
                    f"The checkpoint store holds at most {self.max_checkpoints} checkpoints and "
                    "all of them are in use; raise --max-checkpoints to compare more at once."
                )
            victim = min(unpinned, key=lambda p: (self._read_meta(p) or {}).get("last_used_ns", 0))
            self._announce(
                f"Evicting {victim.relative_to(self.root).parent} to stay within "
                f"{self.max_checkpoints} stored checkpoints"
            )
            self._remove(victim)
            others.remove(victim)

    # -- use ----------------------------------------------------------------------------
    @contextmanager
    def use(self, remote: RemoteFile) -> Iterator[Path]:
        """Yield a verified local copy, downloading it if needed; pinned while in use."""
        path = self.path_for(remote.spec)
        self._pinned.add(path)
        try:
            if self.lookup(remote) is None:
                if path.is_file():  # same revision, different content (branch moved)
                    self._remove(path)
                self._make_room(path)
                self._announce(f"Downloading {remote.spec} ({remote.size / 1e6:.1f} MB)")
                stats = self._download(remote, path)
                rate = stats.bytes_downloaded / stats.seconds / 1e6 if stats.seconds else 0.0
                self._announce(
                    f"Downloaded {remote.spec} in {stats.seconds:.1f}s ({rate:.1f} MB/s)"
                    + (f", resumed from byte {stats.resumed_from:,}" if stats.resumed_from else "")
                )
            self._write_meta(path, remote)  # also refreshes last-use time
            yield path
        finally:
            self._pinned.discard(path)
