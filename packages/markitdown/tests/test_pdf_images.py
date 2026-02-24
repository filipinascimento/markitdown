#!/usr/bin/env python3 -m pytest
"""Tests for optional PDF image extraction and markdown references."""

import os
from pathlib import Path

import pytest

from markitdown import MarkItDown

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")


def test_pdf_image_extraction_and_markdown_links(tmp_path: Path):
    """When enabled, PDF images should be extracted and referenced in markdown."""
    pdf_path = os.path.join(TEST_FILES_DIR, "MEDRPT-2024-PAT-3847_medical_report_scan.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test file not found: {pdf_path}")

    image_dir = tmp_path / "pdf_images"
    result = MarkItDown().convert(
        pdf_path,
        extract_pdf_images=True,
        pdf_image_dir=str(image_dir),
    )

    extracted_files = sorted(path for path in image_dir.iterdir() if path.is_file())
    assert len(extracted_files) > 0, "Expected at least one extracted image file"
    assert "## Extracted Images" in result.markdown

    for index, image_file in enumerate(extracted_files, start=1):
        assert f"![PDF image {index}](" in result.markdown
        assert image_file.name in result.markdown
