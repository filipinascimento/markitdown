#!/usr/bin/env python3 -m pytest
"""Unit tests for PDF figure caption/region helpers."""

from markitdown.converters._pdf_converter import (
    _extract_caption_lines,
    _merge_bboxes,
    _select_region_for_caption,
)


class _FakePage:
    def __init__(self, words):
        self._words = words

    def extract_words(self, **kwargs):
        return self._words


def test_extract_caption_lines_detects_figure_prefixes():
    page = _FakePage(
        [
            {"text": "Figure", "x0": 50, "x1": 90, "top": 100, "bottom": 110},
            {"text": "2", "x0": 95, "x1": 101, "top": 100, "bottom": 110},
            {"text": "A", "x0": 104, "x1": 108, "top": 100, "bottom": 110},
            {"text": "A", "x0": 112, "x1": 118, "top": 100, "bottom": 110},
            {"text": "caption", "x0": 121, "x1": 160, "top": 100, "bottom": 110},
        ]
    )

    captions = _extract_caption_lines(page)
    assert len(captions) == 1
    assert captions[0]["figure_id"] == "2"
    assert captions[0]["text"].startswith("Figure 2")


def test_merge_bboxes_merges_close_regions():
    boxes = [
        (10.0, 10.0, 40.0, 40.0),
        (42.0, 11.0, 80.0, 41.0),  # within merge gap
        (200.0, 50.0, 240.0, 90.0),
    ]
    merged = _merge_bboxes(boxes, gap=5.0)
    assert len(merged) == 2


def test_select_region_for_caption_prefers_nearest_overlap():
    caption = {"x0": 40.0, "x1": 160.0, "top": 500.0}
    regions = [
        (35.0, 380.0, 180.0, 460.0),  # nearest suitable region
        (35.0, 150.0, 180.0, 300.0),
    ]
    selected = _select_region_for_caption(
        caption,
        regions,
        page_width=600.0,
        used_indices=set(),
    )
    assert selected is not None
    assert selected[0] == 0
