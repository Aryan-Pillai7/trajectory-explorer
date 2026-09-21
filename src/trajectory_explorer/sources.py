"""Turn a CLI argument into a checkpoint source: a local file/directory or a Hub revision.

Rules (in order):
1. An existing local path wins, even if it looks like "org/name@revision".
2. A Windows path (``C:\\x``, ``C:/x``, ``\\\\server\\share``) is never a repo spec. It cannot
   exist inside the container, so it gets an error pointing at the /data mount.
3. ``org/name@revision`` resolves through the Hub (metadata only; the file is downloaded
   lazily, through the rolling store, when a pair actually has to be measured).
4. Anything else with an "@" is an unknown format; everything else is a (missing) local path.
"""

from __future__ import annotations

import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from trajectory_explorer import hub
from trajectory_explorer.diff import LocalSource, SourceInfo, local_source
from trajectory_explorer.errors import InputError
from trajectory_explorer.hub import HubSpec, RemoteFile
from trajectory_explorer.store import CheckpointStore

_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_HUB_SPEC = re.compile(
    r"^(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)@(?P<revision>[A-Za-z0-9][\w./-]*)$"
)


def parse_spec(arg: str) -> HubSpec | Path:
    """Classify a CLI argument without touching the network."""
    if not arg:
        raise InputError("Empty checkpoint argument")
    path = Path(arg)
    if path.exists():
        return path
    if _WINDOWS_PATH.match(arg):
        host = os.environ.get("TE_HOST_DATA_DIR", "the data folder")
        raise InputError(
            f"{arg} looks like a Windows path, but the tool runs inside Docker and only sees "
            f"/data (which is {host} on your machine). Put the checkpoint under that folder and "
            "pass its /data/... path, or use a Hub source like EleutherAI/pythia-70m@step1000."
        )
    match = _HUB_SPEC.match(arg)
    if match:
        return HubSpec(match["repo"], match["revision"])
    if "@" in arg:
        raise InputError(
            f"{arg!r} is neither an existing local path nor a Hub source of the form "
            "org/name@revision (e.g. EleutherAI/pythia-70m@step1000)."
        )
    return path


@dataclass(frozen=True)
class HubSource:
    remote: RemoteFile
    store: CheckpointStore
    info: SourceInfo

    def fetch(self) -> AbstractContextManager[Path]:
        return self.store.use(self.remote)

    def needs_download(self) -> bool:
        return self.store.lookup(self.remote) is None


def hub_source(spec: HubSpec, store: CheckpointStore) -> HubSource:
    """Look up size and LFS sha256 on the Hub (no download)."""
    remote = hub.fetch_metadata(spec)
    name = spec.repo.split("/", 1)[1]
    info = SourceInfo(
        label=f"{name}@{spec.revision}",
        path=str(spec),
        sha256=remote.sha256,
        size_bytes=remote.size,
    )
    return HubSource(remote, store, info)


def resolve(arg: str, store: CheckpointStore) -> LocalSource | HubSource:
    """A CLI argument -> a source whose content hash is known before any download."""
    spec = parse_spec(arg)
    if isinstance(spec, HubSpec):
        return hub_source(spec, store)
    if spec.is_dir():
        candidate = spec / hub.FILENAME
        if not candidate.is_file():
            raise InputError(f"Directory {arg} has no {hub.FILENAME}")
        spec = candidate
    return local_source(spec)
