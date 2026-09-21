"""Placeholder unit test until the real suite lands; later folded into the exit-code test."""

import pytest

from trajectory_explorer import __version__
from trajectory_explorer.cli import main


@pytest.mark.unit
def test_version_flag_prints_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"trajectory-explorer {__version__}"
