"""An in-process fake Hugging Face Hub for tests (no real network traffic).

Two HTTP servers on 127.0.0.1: the "hub" (metadata API + /resolve/ URLs that redirect with
302) and a "cdn" on a different port (so it is a different host for redirect purposes) that
serves the bytes with Range support. The response shapes copy the real Hub, as checked on
2026-09-22 (decision D32). Behaviours can be switched per (repo, revision): wrong bytes,
a dropped connection after N bytes (once), .bin-only revisions, and a response delay.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

Key = tuple[str, str]  # (repo, revision)


class FakeHub:
    def __init__(self) -> None:
        self.files: dict[Key, bytes] = {}
        self.bin_only: set[Key] = set()
        self.corrupt: set[Key] = set()
        self.drop_after: dict[Key, int] = {}
        self.delay = 0.0
        self.requests: list[dict] = []  # {"server", "method", "path", "headers"}
        self._servers = [self._serve("hub"), self._serve("cdn")]

    # -- public -----------------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._servers[0].server_address[1]}"

    @property
    def cdn_url(self) -> str:
        return f"http://127.0.0.1:{self._servers[1].server_address[1]}"

    def add(self, repo: str, revision: str, data: bytes) -> str:
        self.files[(repo, revision)] = data
        return hashlib.sha256(data).hexdigest()

    def cdn_gets(self) -> list[str]:
        """Revisions whose bytes were requested from the CDN, in order."""
        return [r["path"].split("/")[-1] for r in self.requests if r["server"] == "cdn"]

    def close(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()

    # -- server -------------------------------------------------------------------------
    def _serve(self, role: str) -> ThreadingHTTPServer:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep test output quiet
                pass

            def do_GET(self) -> None:
                hub.requests.append(
                    {
                        "server": role,
                        "method": "GET",
                        "path": self.path,
                        "headers": dict(self.headers),
                    }
                )
                if hub.delay:
                    time.sleep(hub.delay)
                parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/")]
                if role == "hub" and parts[:2] == ["api", "models"] and len(parts) == 6:
                    self._tree(f"{parts[2]}/{parts[3]}", parts[5])
                elif role == "hub" and len(parts) == 5 and parts[2] == "resolve":
                    self._resolve(f"{parts[0]}/{parts[1]}", parts[3], parts[4])
                elif role == "cdn" and len(parts) == 4 and parts[0] == "cdn":
                    self._bytes((f"{parts[1]}/{parts[2]}", parts[3]))
                else:
                    self._json(404, {"error": "not found"})

            def _json(self, code: int, body: object) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _tree(self, repo: str, revision: str) -> None:
                known = {r for r, _ in hub.files} | {r for r, _ in hub.bin_only}
                if repo not in known:
                    return self._json(401, {"error": "Invalid username or password."})
                key = (repo, revision)
                if key in hub.bin_only:
                    entries = [{"type": "file", "path": "pytorch_model.bin", "size": 10}]
                elif key in hub.files:
                    data = hub.files[key]
                    lfs = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
                    entries = [
                        {"type": "file", "path": "config.json", "size": 2, "oid": "0" * 40},
                        {
                            "type": "file",
                            "path": "model.safetensors",
                            "size": len(data),
                            "oid": "1" * 40,
                            "lfs": {**lfs, "pointerSize": 134},
                        },
                    ]
                else:
                    return self._json(404, {"error": f"Invalid rev id: {revision}"})
                self._json(200, entries)

            def _resolve(self, repo: str, revision: str, filename: str) -> None:
                if (repo, revision) not in hub.files or filename != "model.safetensors":
                    return self._json(404, {"error": "Entry not found"})
                self.send_response(302)
                self.send_header("Location", f"{hub.cdn_url}/cdn/{repo}/{revision}")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _bytes(self, key: Key) -> None:
                data = hub.files[key]
                if key in hub.corrupt:
                    data = bytes([data[0] ^ 0xFF]) + data[1:]
                start = 0
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes=") and rng.endswith("-"):
                    start = int(rng[len("bytes=") : -1])
                body = data[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                cut = hub.drop_after.pop(key, None)
                if cut is not None:  # send part of the body, then drop the connection
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()
        return server
