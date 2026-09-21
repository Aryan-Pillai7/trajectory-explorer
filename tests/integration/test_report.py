import shutil
from html.parser import HTMLParser

import numpy as np
import pytest

from conftest import neox_tensors, perturb
from trajectory_explorer.diff import diff_checkpoints
from trajectory_explorer.report import render_html


class ReportParser(HTMLParser):
    """Collects the structure the tests check, using only the stdlib parser."""

    def __init__(self) -> None:
        super().__init__()
        self.section_ids: list[str] = []
        self.sig_cells = 0
        self.ranked_rows = 0  # rows of the default (significant-only) ranked table
        self.banner = ""
        self.external_refs: list[str] = []
        self.forbidden_tags: list[str] = []
        self._in_banner = False
        self._section = ""
        self._details_depth = 0
        self._in_tbody = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for value in a.values():
            if value and ("http://" in value or "https://" in value):
                self.external_refs.append(value)
        if tag in {"script", "link", "img", "iframe", "object", "embed"}:
            self.forbidden_tags.append(tag)
        if a.get("id") == "summary":
            self._in_banner = True
        if tag == "section":
            self._section = a.get("id", "")
            self.section_ids.append(self._section)
        if tag == "g" and "cell sig" in a.get("class", ""):
            self.sig_cells += 1
        if tag == "details":
            self._details_depth += 1
        if tag == "tbody":
            self._in_tbody = True
        if tag == "tr" and self._in_tbody and self._section == "ranked" and not self._details_depth:
            self.ranked_rows += 1

    def handle_endtag(self, tag):
        if tag == "div" and self._in_banner:
            self._in_banner = False
        if tag == "details":
            self._details_depth -= 1
        if tag == "tbody":
            self._in_tbody = False

    def handle_data(self, data):
        if self._in_banner:
            self.banner += data


def parse(html: str) -> ReportParser:
    parser = ReportParser()
    parser.feed(html)
    parser.close()
    return parser


@pytest.mark.integration
def test_report_has_sections_in_order_is_offline_and_ranks_changes(make_checkpoint, rng):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", perturb(base, 0.01, rng))
    html = render_html(diff_checkpoints(a, b))
    report = parse(html)

    assert html.startswith("<!DOCTYPE html>")
    assert report.section_ids == ["heatmap", "ranked", "noise-floor", "labels"]
    assert html.index('id="summary"') < html.index('id="heatmap"')
    assert report.banner.startswith("The largest change is in")
    assert report.sig_cells > 0
    assert report.ranked_rows == len(base)  # every tensor moved 1%: all significant
    assert report.external_refs == []
    assert report.forbidden_tags == []


@pytest.mark.integration
@pytest.mark.parametrize("null_pair", ["byte_identical", "float16_rounding_only"])
def test_null_pairs_render_no_significant_difference(null_pair, make_checkpoint, rng, tmp_path):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    if null_pair == "byte_identical":
        b = tmp_path / "b.safetensors"
        shutil.copyfile(a, b)
    else:
        b = make_checkpoint("b.safetensors", {n: t.astype(np.float16) for n, t in base.items()})

    report = parse(render_html(diff_checkpoints(a, b)))

    assert report.banner.startswith("No significant difference")
    assert report.sig_cells == 0
    assert report.ranked_rows == 0
    assert report.section_ids == ["heatmap", "ranked", "noise-floor", "labels"]
