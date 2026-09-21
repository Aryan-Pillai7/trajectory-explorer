"""Hugging Face Hub access with the standard library only (no huggingface_hub, decision D4).

Verified against the real Hub on 2026-09-22 (decision D32):
- ``GET {endpoint}/api/models/{repo}/tree/{revision}`` lists the revision's root files; the
  model.safetensors entry carries ``lfs.oid`` (bare sha256 hex of the file bytes) and ``lfs.size``.
- ``GET {endpoint}/{repo}/resolve/{revision}/model.safetensors`` answers 302 to a CDN on another
  host, which honours ``Range`` (206 + Content-Range), so downloads can resume.
- Unknown repo: 401 (the Hub does not reveal whether a repo exists). Unknown revision: 404
  "Invalid rev id". Missing file: 404 with X-Error-Code EntryNotFound.

An optional HF_TOKEN is read from the environment only, sent as a Bearer header to the Hub
host, dropped on any redirect to a different host, and never logged or written anywhere.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from trajectory_explorer.errors import InputError

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
FILENAME = "model.safetensors"
DEFAULT_TIMEOUT = 30.0  # seconds, per connect and per socket read
_CHUNK = 1024 * 1024
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HubSpec:
    repo: str  # "org/name"
    revision: str  # branch, tag or commit, e.g. "step1000"

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}"


@dataclass(frozen=True)
class RemoteFile:
    spec: HubSpec
    size: int
    sha256: str  # LFS oid: sha256 of the file bytes, same as a local sha256_file()


@dataclass(frozen=True)
class DownloadStats:
    bytes_downloaded: int
    seconds: float
    resumed_from: int


def endpoint() -> str:
    """The Hub base URL. HF_ENDPOINT overrides it (used by the tests' fake Hub)."""
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never forward the Authorization header to another host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def _request(url: str, headers: dict[str, str], timeout: float) -> http.client.HTTPResponse:
    """Open a URL. HTTP errors propagate as HTTPError; network problems become InputError."""
    req = urllib.request.Request(url, headers=headers)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    host = urlsplit(url).netloc
    opener = urllib.request.build_opener(_NoCrossHostAuth)
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError | socket.timeout):
            raise InputError(f"Timed out after {timeout:g}s contacting {host}") from None
        raise InputError(f"Network error contacting {host}: {exc.reason}") from None
    except TimeoutError:
        raise InputError(f"Timed out after {timeout:g}s contacting {host}") from None
    except OSError as exc:
        raise InputError(f"Network error contacting {host}: {exc}") from None


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8", "replace"))
        return str(body.get("error", "")) if isinstance(body, dict) else ""
    except (ValueError, OSError):
        return ""


def fetch_metadata(spec: HubSpec, *, timeout: float | None = None) -> RemoteFile:
    """Size and LFS sha256 of model.safetensors at a revision, without downloading it."""
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    url = f"{endpoint()}/api/models/{spec.repo}/tree/{quote(spec.revision, safe='')}"
    try:
        with _request(url, {"Accept": "application/json"}, timeout) as resp:
            entries = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _error_text(exc)
        if exc.code in (401, 403):
            raise InputError(
                f"Repository {spec.repo} was not found on the Hub, or it is private or gated "
                "(set HF_TOKEN in .env for gated models)."
            ) from None
        if exc.code == 404:
            raise InputError(
                f"Revision {spec.revision!r} was not found in {spec.repo}"
                + (f" ({detail})." if detail else ".")
            ) from None
        raise InputError(
            f"The Hub returned HTTP {exc.code} for {spec}: {detail or exc.reason}"
        ) from None
    except (ValueError, TimeoutError) as exc:
        raise InputError(f"Could not read Hub metadata for {spec}: {exc}") from None

    files = {e.get("path"): e for e in entries if isinstance(e, dict)}
    entry = files.get(FILENAME)
    if entry is None:
        only_bin = "pytorch_model.bin" in files
        raise InputError(
            f"{spec} has no {FILENAME}"
            + (" (it only ships pytorch_model.bin)" if only_bin else "")
            + ". Converting .bin checkpoints needs a one-time torch conversion step (the D1 "
            "contingency), which is not built."
        )
    lfs = entry.get("lfs") or {}
    sha, size = str(lfs.get("oid", "")), lfs.get("size", entry.get("size"))
    if not _SHA256_HEX.match(sha) or not isinstance(size, int):
        raise InputError(f"The Hub metadata for {spec} has no usable LFS sha256/size.")
    return RemoteFile(spec=spec, size=size, sha256=sha)


def list_branches(repo: str, *, timeout: float | None = None) -> list[str]:
    """Branch names of a model repo (``GET /api/models/{repo}/refs``; verified, D37)."""
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    try:
        with _request(f"{endpoint()}/api/models/{repo}/refs", {}, timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise InputError(
                f"Repository {repo} was not found on the Hub, or it is private or gated "
                "(set HF_TOKEN in .env for gated models)."
            ) from None
        raise InputError(f"The Hub returned HTTP {exc.code} listing branches of {repo}") from None
    except (ValueError, TimeoutError) as exc:
        raise InputError(f"Could not read the branch list of {repo}: {exc}") from None
    branches = data.get("branches", []) if isinstance(data, dict) else []
    return [b["name"] for b in branches if isinstance(b, dict) and isinstance(b.get("name"), str)]


def download(
    remote: RemoteFile,
    dest: Path,
    *,
    timeout: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DownloadStats:
    """Download to ``dest`` via ``dest.part``: resume with Range, hash while streaming, verify.

    The file is renamed into place only after its size and sha256 match the Hub's LFS
    metadata. On a mismatch the .part file is deleted. An interrupted download keeps the .part
    file so the next run resumes it.
    """
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    offset = 0
    if part.exists():
        if part.stat().st_size > remote.size:
            part.unlink()
        else:  # re-hash what we already have, so the final digest covers the whole file
            with part.open("rb") as fh:
                while chunk := fh.read(_CHUNK):
                    hasher.update(chunk)
            offset = part.stat().st_size
    resumed_from = offset

    start = time.perf_counter()
    received = 0
    if offset < remote.size:
        revision = quote(remote.spec.revision, safe="")
        url = f"{endpoint()}/{remote.spec.repo}/resolve/{revision}/{FILENAME}"
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            resp = _request(url, headers, timeout)
        except urllib.error.HTTPError as exc:
            reason = _error_text(exc) or exc.reason
            raise InputError(
                f"Download of {remote.spec} failed: HTTP {exc.code} {reason}"
            ) from None
        if offset and resp.status == 200:  # server ignored Range: start over
            log.info("Server ignored the Range request; restarting %s from byte 0", remote.spec)
            hasher, offset, resumed_from = hashlib.sha256(), 0, 0
        elif offset and not resp.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            resp.close()
            raise InputError(f"Unexpected Content-Range while resuming {remote.spec}")
        try:
            with resp, part.open("ab" if offset else "wb") as out:
                while chunk := resp.read(_CHUNK):
                    out.write(chunk)
                    hasher.update(chunk)
                    received += len(chunk)
                    if progress:
                        progress(offset + received, remote.size)
        except (http.client.HTTPException, OSError) as exc:
            raise InputError(
                f"Download of {remote.spec} was interrupted after {offset + received:,} of "
                f"{remote.size:,} bytes ({type(exc).__name__}). Run the command again to resume."
            ) from None

    size = part.stat().st_size
    if size < remote.size:
        raise InputError(
            f"Download of {remote.spec} stopped at {size:,} of {remote.size:,} bytes. "
            "Run the command again to resume."
        )
    digest = hasher.hexdigest()
    if size != remote.size or digest != remote.sha256:
        part.unlink()
        raise InputError(
            f"Download of {remote.spec} failed verification: expected sha256 "
            f"{remote.sha256[:12]}... ({remote.size:,} bytes), got {digest[:12]}... "
            f"({size:,} bytes). The partial file was removed."
        )
    os.replace(part, dest)
    return DownloadStats(received, time.perf_counter() - start, resumed_from)
