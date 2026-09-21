"""Opt-in test against the real Hugging Face Hub. Downloads about 450 MB.

Run explicitly:  docker compose run --rm test -m network
Data goes to $TE_NETWORK_DATA_DIR (default /data/tmp/network-test, i.e. under TE_DATA_DIR on
the host), not to the container's /tmp.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from trajectory_explorer.cli import main


@pytest.mark.network
@pytest.mark.slow
def test_pythia_main_is_a_float16_copy_of_step143000(monkeypatch, tmp_path):
    data = Path(os.environ.get("TE_NETWORK_DATA_DIR", "/data/tmp/network-test"))
    monkeypatch.setenv("TE_DATA_DIR", str(data))
    monkeypatch.delenv("HF_ENDPOINT", raising=False)  # undo the offline default: real Hub
    js = tmp_path / "result.json"
    try:
        code = main(
            [
                "diff",
                "EleutherAI/pythia-70m@step143000",
                "EleutherAI/pythia-70m@main",
                "--json",
                str(js),
                "-o",
                str(tmp_path / "report.html"),
            ]
        )
        assert code == 0
        result = json.loads(js.read_text(encoding="utf-8"))
        assert result["n_tensors"] == 76 and len(result["skipped"]) == 18
        assert not any(m["status"] == "significant" for m in result["tensors"].values())
    finally:
        shutil.rmtree(data / "checkpoints", ignore_errors=True)
