import sys
import io
import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Any

from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE

# Pattern for MasterFormat-style partial numbering (e.g., ".1", ".2", ".10")
PARTIAL_NUMBERING_PATTERN = re.compile(r"^\.\d+$")
FIGURE_CAPTION_PATTERN = re.compile(r"^\s*(fig(?:ure)?\.?)\s*(\d+[A-Za-z]?)\b", re.I)


def _merge_partial_numbering_lines(text: str) -> str:
    """
    Post-process extracted text to merge MasterFormat-style partial numbering
    with the following text line.

    MasterFormat documents use partial numbering like:
        .1  The intent of this Request for Proposal...
        .2  Available information relative to...

    Some PDF extractors split these into separate lines:
        .1
        The intent of this Request for Proposal...

    This function merges them back together.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line is ONLY a partial numbering
        if PARTIAL_NUMBERING_PATTERN.match(stripped):
            # Look for the next non-empty line to merge with
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                # Merge the partial numbering with the next line
                next_line = lines[j].strip()
                result_lines.append(f"{stripped} {next_line}")
                i = j + 1  # Skip past the merged line
            else:
                # No next line to merge with, keep as is
                result_lines.append(line)
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


# Load dependencies
_dependency_exc_info = None
try:
    import pdfminer
    import pdfminer.high_level
    import pdfplumber
except ImportError:
    _dependency_exc_info = sys.exc_info()


ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/pdf",
    "application/x-pdf",
]

ACCEPTED_FILE_EXTENSIONS = [".pdf"]


def _to_markdown_table(table: list[list[str]], include_separator: bool = True) -> str:
    """Convert a 2D list (rows/columns) into a nicely aligned Markdown table.

    Args:
        table: 2D list of cell values
        include_separator: If True, include header separator row (standard markdown).
                          If False, output simple pipe-separated rows.
    """
    if not table:
        return ""

    # Normalize None → ""
    table = [[cell if cell is not None else "" for cell in row] for row in table]

    # Filter out empty rows
    table = [row for row in table if any(cell.strip() for cell in row)]

    if not table:
        return ""

    # Column widths
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    def fmt_row(row: list[str]) -> str:
        return (
            "|"
            + "|".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
            + "|"
        )

    if include_separator:
        header, *rows = table
        md = [fmt_row(header)]
        md.append("|" + "|".join("-" * w for w in col_widths) + "|")
        for row in rows:
            md.append(fmt_row(row))
    else:
        md = [fmt_row(row) for row in table]

    return "\n".join(md)


def _sanitize_stem(name: str) -> str:
    """Normalize names for filesystem-safe file/directory output."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "document"


def _resolve_pdf_image_paths(
    stream_info: StreamInfo, pdf_image_dir: str | None
) -> tuple[Path, Path]:
    """Return absolute output directory and markdown link directory."""
    source_name = stream_info.filename or stream_info.local_path or "document.pdf"
    source_stem = _sanitize_stem(Path(source_name).stem)

    if pdf_image_dir is not None:
        requested = Path(pdf_image_dir).expanduser()
        if requested.is_absolute():
            return requested, requested
        return (Path.cwd() / requested).resolve(), requested

    default_dir = Path(f"{source_stem}_images")
    return (Path.cwd() / default_dir).resolve(), default_dir


def _extract_pdf_images(
    pdf_bytes: io.BytesIO,
    *,
    stream_info: StreamInfo,
    pdf_image_dir: str | None,
) -> list[str]:
    """
    Extract embedded image objects from a PDF and return markdown-safe link paths.
    """
    output_dir, link_dir = _resolve_pdf_image_paths(stream_info, pdf_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = stream_info.filename or stream_info.local_path or "document.pdf"
    source_stem = _sanitize_stem(Path(source_name).stem)

    with tempfile.TemporaryDirectory(prefix="markitdown_pdf_images_") as temp_dir:
        pdf_bytes.seek(0)
        pdfminer.high_level.extract_text_to_fp(
            pdf_bytes,
            io.StringIO(),
            output_type="text",
            codec="utf-8",
            output_dir=temp_dir,
        )

        extracted = sorted(
            path for path in Path(temp_dir).iterdir() if path.is_file()
        )

        image_links: list[str] = []
        for idx, src in enumerate(extracted, start=1):
            suffix = src.suffix.lower() or ".bin"
            base_name = f"{source_stem}_image_{idx}{suffix}"
            dst = output_dir / base_name

            # Avoid overwriting existing files from prior runs.
            if dst.exists():
                collision = 1
                while True:
                    candidate = output_dir / f"{source_stem}_image_{idx}_{collision}{suffix}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    collision += 1

            shutil.copyfile(src, dst)

            if link_dir.is_absolute():
                image_links.append(dst.as_posix())
            else:
                image_links.append((link_dir / dst.name).as_posix())

        return image_links


def _resolve_pdf_figure_paths(
    stream_info: StreamInfo, pdf_figure_dir: str | None
) -> tuple[Path, Path]:
    """Return absolute output directory and markdown link directory for figures."""
    source_name = stream_info.filename or stream_info.local_path or "document.pdf"
    source_stem = _sanitize_stem(Path(source_name).stem)

    if pdf_figure_dir is not None:
        requested = Path(pdf_figure_dir).expanduser()
        if requested.is_absolute():
            return requested, requested
        return (Path.cwd() / requested).resolve(), requested

    default_dir = Path(f"{source_stem}_figures")
    return (Path.cwd() / default_dir).resolve(), default_dir


def _to_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Normalize layout objects to an (x0, top, x1, bottom) tuple."""
    try:
        x0 = float(obj["x0"])
        x1 = float(obj["x1"])
        top = float(obj["top"])
        bottom = float(obj["bottom"])
    except (KeyError, TypeError, ValueError):
        return None

    if x1 <= x0 or bottom <= top:
        return None

    return (x0, top, x1, bottom)


def _merge_bboxes(
    bboxes: list[tuple[float, float, float, float]],
    *,
    gap: float = 10.0,
) -> list[tuple[float, float, float, float]]:
    """Merge overlapping/nearby boxes to build coherent graphic regions."""
    current = sorted(bboxes, key=lambda b: (b[1], b[0]))
    while True:
        merged: list[tuple[float, float, float, float]] = []
        for box in current:
            x0, top, x1, bottom = box
            did_merge = False
            for i, (mx0, mtop, mx1, mbottom) in enumerate(merged):
                intersects = not (
                    x1 < mx0 - gap
                    or x0 > mx1 + gap
                    or bottom < mtop - gap
                    or top > mbottom + gap
                )
                if intersects:
                    merged[i] = (
                        min(x0, mx0),
                        min(top, mtop),
                        max(x1, mx1),
                        max(bottom, mbottom),
                    )
                    did_merge = True
                    break

            if not did_merge:
                merged.append(box)

        if merged == current:
            return merged
        current = sorted(merged, key=lambda b: (b[1], b[0]))


def _extract_caption_lines(page: Any) -> list[dict[str, Any]]:
    """Extract figure captions (e.g., Fig. 1 / Figure 2) as line-level records."""
    words = page.extract_words(keep_blank_chars=True, x_tolerance=2, y_tolerance=2)
    if not words:
        return []

    y_tolerance = 3
    lines_by_y: dict[float, list[dict[str, Any]]] = {}
    for word in words:
        y_key = round(float(word["top"]) / y_tolerance) * y_tolerance
        lines_by_y.setdefault(y_key, []).append(word)

    captions: list[dict[str, Any]] = []
    for y_key in sorted(lines_by_y.keys()):
        row_words = sorted(lines_by_y[y_key], key=lambda w: float(w["x0"]))
        text = " ".join(str(w["text"]).strip() for w in row_words).strip()
        if not text:
            continue

        match = FIGURE_CAPTION_PATTERN.match(text)
        if match is None:
            continue

        captions.append(
            {
                "text": text,
                "figure_id": match.group(2),
                "x0": min(float(w["x0"]) for w in row_words),
                "x1": max(float(w["x1"]) for w in row_words),
                "top": min(float(w["top"]) for w in row_words),
                "bottom": max(float(w["bottom"]) for w in row_words),
            }
        )

    return captions


def _collect_graphic_regions(page: Any) -> list[tuple[float, float, float, float]]:
    """Collect non-text layout element regions likely belonging to figures."""
    raw_boxes: list[tuple[float, float, float, float]] = []

    for element in getattr(page, "images", []):
        bbox = _to_bbox(element)
        if bbox is not None:
            raw_boxes.append(bbox)

    for element in getattr(page, "rects", []):
        bbox = _to_bbox(element)
        if bbox is not None:
            raw_boxes.append(bbox)

    for element in getattr(page, "curves", []):
        bbox = _to_bbox(element)
        if bbox is not None:
            raw_boxes.append(bbox)

    for element in getattr(page, "lines", []):
        bbox = _to_bbox(element)
        if bbox is not None:
            raw_boxes.append(bbox)

    # Ignore tiny decorative artifacts.
    filtered = []
    for x0, top, x1, bottom in raw_boxes:
        width = x1 - x0
        height = bottom - top
        if width >= 6 and height >= 6:
            filtered.append((x0, top, x1, bottom))

    return _merge_bboxes(filtered, gap=12.0)


def _horizontal_overlap_ratio(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    denom = max(1.0, min(a[1] - a[0], b[1] - b[0]))
    return overlap / denom


def _expand_region_with_nearby_words(
    region: tuple[float, float, float, float],
    words: list[dict[str, Any]],
    *,
    caption_top: float,
    page_width: float,
    margin: float = 18.0,
) -> tuple[float, float, float, float]:
    """Expand a graphic region to include nearby labels (axes/ticks/legends)."""
    x0, top, x1, bottom = region
    expanded = (x0, top, x1, bottom)

    for word in words:
        wbox = _to_bbox(word)
        if wbox is None:
            continue
        wx0, wtop, wx1, wbottom = wbox

        # Keep labels that are spatially near the graphic and above the caption.
        if wbottom > caption_top + 2:
            continue
        if wx1 < expanded[0] - margin or wx0 > expanded[2] + margin:
            continue
        if wbottom < expanded[1] - margin or wtop > expanded[3] + margin:
            continue

        expanded = (
            min(expanded[0], wx0),
            min(expanded[1], wtop),
            max(expanded[2], wx1),
            max(expanded[3], wbottom),
        )

    return (
        max(0.0, expanded[0] - margin),
        max(0.0, expanded[1] - margin),
        min(page_width, expanded[2] + margin),
        expanded[3] + margin,
    )


def _select_region_for_caption(
    caption: dict[str, Any],
    regions: list[tuple[float, float, float, float]],
    *,
    page_width: float,
    used_indices: set[int],
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Choose the most plausible region above a figure caption."""
    caption_top = float(caption["top"])
    caption_x0 = float(caption["x0"])
    caption_x1 = float(caption["x1"])

    best_idx = -1
    best_score = float("inf")
    best_region: tuple[float, float, float, float] | None = None

    caption_band = (
        max(0.0, caption_x0 - page_width * 0.12),
        min(page_width, caption_x1 + page_width * 0.12),
    )

    def choose(min_overlap: float) -> tuple[int, float, tuple[float, float, float, float]] | None:
        local_best_idx = -1
        local_best_score = float("inf")
        local_best_region: tuple[float, float, float, float] | None = None

        for idx, region in enumerate(regions):
            if idx in used_indices:
                continue

            rx0, rtop, rx1, rbottom = region
            width = rx1 - rx0
            height = rbottom - rtop
            if width < page_width * 0.12 or height < 24:
                continue

            if rbottom > caption_top - 2:
                continue

            vertical_gap = caption_top - rbottom
            if vertical_gap > 520:
                continue

            overlap = _horizontal_overlap_ratio((rx0, rx1), caption_band)
            if overlap < min_overlap:
                continue

            area_penalty = 4000.0 / max(1.0, width * height)
            score = vertical_gap + (1.0 - overlap) * 120 + area_penalty
            if score < local_best_score:
                local_best_idx = idx
                local_best_score = score
                local_best_region = region

        if local_best_region is None:
            return None
        return local_best_idx, local_best_score, local_best_region

    strict = choose(0.18)
    relaxed = choose(0.08) if strict is None else strict
    if relaxed is not None:
        best_idx, best_score, best_region = relaxed

    if best_region is None:
        return None
    return best_idx, best_region


def _extract_pdf_figures(
    pdf_bytes: io.BytesIO,
    *,
    stream_info: StreamInfo,
    pdf_figure_dir: str | None,
    pdf_figure_dpi: int = 300,
) -> list[tuple[str, str]]:
    """
    Extract rendered figure crops by aligning caption lines with nearby graphic regions.
    """
    output_dir, link_dir = _resolve_pdf_figure_paths(stream_info, pdf_figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = stream_info.filename or stream_info.local_path or "document.pdf"
    source_stem = _sanitize_stem(Path(source_name).stem)
    figure_dpi = max(96, int(pdf_figure_dpi))

    results: list[tuple[str, str]] = []
    figure_counter = 0

    pdf_bytes.seek(0)
    with pdfplumber.open(pdf_bytes) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            captions = _extract_caption_lines(page)
            if not captions:
                continue

            regions = _collect_graphic_regions(page)
            if not regions:
                continue

            words = page.extract_words(keep_blank_chars=True, x_tolerance=2, y_tolerance=2)
            used_indices: set[int] = set()
            page_image = None

            for caption in captions:
                selected = _select_region_for_caption(
                    caption,
                    regions,
                    page_width=float(page.width),
                    used_indices=used_indices,
                )
                if selected is None:
                    continue

                region_idx, base_region = selected
                used_indices.add(region_idx)
                figure_region = _expand_region_with_nearby_words(
                    base_region,
                    words,
                    caption_top=float(caption["top"]),
                    page_width=float(page.width),
                )

                if page_image is None:
                    page_image = page.to_image(resolution=figure_dpi).original

                scale = figure_dpi / 72.0
                left = int(max(0.0, figure_region[0] * scale))
                top = int(max(0.0, figure_region[1] * scale))
                right = int(min(page_image.width, figure_region[2] * scale))
                bottom = int(min(page_image.height, figure_region[3] * scale))

                if right - left < 30 or bottom - top < 30:
                    continue

                crop = page_image.crop((left, top, right, bottom))
                figure_counter += 1
                figure_id = str(caption["figure_id"]).replace(" ", "")
                filename = (
                    f"{source_stem}_figure_{figure_id}.png"
                    if figure_id
                    else f"{source_stem}_figure_{figure_counter}.png"
                )
                destination = output_dir / filename

                if destination.exists():
                    suffix = 1
                    while True:
                        candidate = output_dir / (
                            f"{Path(filename).stem}_{suffix}{Path(filename).suffix}"
                        )
                        if not candidate.exists():
                            destination = candidate
                            break
                        suffix += 1

                crop.save(destination, format="PNG")

                if link_dir.is_absolute():
                    markdown_path = destination.as_posix()
                else:
                    markdown_path = (link_dir / destination.name).as_posix()

                results.append((str(caption["text"]), markdown_path))

    return results


def _extract_form_content_from_words(page: Any) -> str | None:
    """
    Extract form-style content from a PDF page by analyzing word positions.
    This handles borderless forms/tables where words are aligned in columns.

    Returns markdown with proper table formatting:
    - Tables have pipe-separated columns with header separator rows
    - Non-table content is rendered as plain text

    Returns None if the page doesn't appear to be a form-style document,
    indicating that pdfminer should be used instead for better text spacing.
    """
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return None

    # Group words by their Y position (rows)
    y_tolerance = 5
    rows_by_y: dict[float, list[dict]] = {}
    for word in words:
        y_key = round(word["top"] / y_tolerance) * y_tolerance
        if y_key not in rows_by_y:
            rows_by_y[y_key] = []
        rows_by_y[y_key].append(word)

    # Sort rows by Y position
    sorted_y_keys = sorted(rows_by_y.keys())
    page_width = page.width if hasattr(page, "width") else 612

    # First pass: analyze each row
    row_info: list[dict] = []
    for y_key in sorted_y_keys:
        row_words = sorted(rows_by_y[y_key], key=lambda w: w["x0"])
        if not row_words:
            continue

        first_x0 = row_words[0]["x0"]
        last_x1 = row_words[-1]["x1"]
        line_width = last_x1 - first_x0
        combined_text = " ".join(w["text"] for w in row_words)

        # Count distinct x-position groups (columns)
        x_positions = [w["x0"] for w in row_words]
        x_groups: list[float] = []
        for x in sorted(x_positions):
            if not x_groups or x - x_groups[-1] > 50:
                x_groups.append(x)

        # Determine row type
        is_paragraph = line_width > page_width * 0.55 and len(combined_text) > 60

        # Check for MasterFormat-style partial numbering (e.g., ".1", ".2")
        # These should be treated as list items, not table rows
        has_partial_numbering = False
        if row_words:
            first_word = row_words[0]["text"].strip()
            if PARTIAL_NUMBERING_PATTERN.match(first_word):
                has_partial_numbering = True

        row_info.append(
            {
                "y_key": y_key,
                "words": row_words,
                "text": combined_text,
                "x_groups": x_groups,
                "is_paragraph": is_paragraph,
                "num_columns": len(x_groups),
                "has_partial_numbering": has_partial_numbering,
            }
        )

    # Collect ALL x-positions from rows with 3+ columns (table-like rows)
    # This gives us the global column structure
    all_table_x_positions: list[float] = []
    for info in row_info:
        if info["num_columns"] >= 3 and not info["is_paragraph"]:
            all_table_x_positions.extend(info["x_groups"])

    if not all_table_x_positions:
        return None

    # Compute adaptive column clustering tolerance based on gap analysis
    all_table_x_positions.sort()

    # Calculate gaps between consecutive x-positions
    gaps = []
    for i in range(len(all_table_x_positions) - 1):
        gap = all_table_x_positions[i + 1] - all_table_x_positions[i]
        if gap > 5:  # Only significant gaps
            gaps.append(gap)

    # Determine optimal tolerance using statistical analysis
    if gaps and len(gaps) >= 3:
        # Use 70th percentile of gaps as threshold (balances precision/recall)
        sorted_gaps = sorted(gaps)
        percentile_70_idx = int(len(sorted_gaps) * 0.70)
        adaptive_tolerance = sorted_gaps[percentile_70_idx]

        # Clamp tolerance to reasonable range [25, 50]
        adaptive_tolerance = max(25, min(50, adaptive_tolerance))
    else:
        # Fallback to conservative value
        adaptive_tolerance = 35

    # Compute global column boundaries using adaptive tolerance
    global_columns: list[float] = []
    for x in all_table_x_positions:
        if not global_columns or x - global_columns[-1] > adaptive_tolerance:
            global_columns.append(x)

    # Adaptive max column check based on page characteristics
    # Calculate average column width
    if len(global_columns) > 1:
        content_width = global_columns[-1] - global_columns[0]
        avg_col_width = content_width / len(global_columns)

        # Forms with very narrow columns (< 30px) are likely dense text
        if avg_col_width < 30:
            return None

        # Compute adaptive max based on columns per inch
        # Typical forms have 3-8 columns per inch
        columns_per_inch = len(global_columns) / (content_width / 72)

        # If density is too high (> 10 cols/inch), likely not a form
        if columns_per_inch > 10:
            return None

        # Adaptive max: allow more columns for wider pages
        # Standard letter is 612pt wide, so scale accordingly
        adaptive_max_columns = int(20 * (page_width / 612))
        adaptive_max_columns = max(15, adaptive_max_columns)  # At least 15

        if len(global_columns) > adaptive_max_columns:
            return None
    else:
        # Single column, not a form
        return None

    # Now classify each row as table row or not
    # A row is a table row if it has words that align with 2+ of the global columns
    for info in row_info:
        if info["is_paragraph"]:
            info["is_table_row"] = False
            continue

        # Rows with partial numbering (e.g., ".1", ".2") are list items, not table rows
        if info["has_partial_numbering"]:
            info["is_table_row"] = False
            continue

        # Count how many global columns this row's words align with
        aligned_columns: set[int] = set()
        for word in info["words"]:
            word_x = word["x0"]
            for col_idx, col_x in enumerate(global_columns):
                if abs(word_x - col_x) < 40:
                    aligned_columns.add(col_idx)
                    break

        # If row uses 2+ of the established columns, it's a table row
        info["is_table_row"] = len(aligned_columns) >= 2

    # Find table regions (consecutive table rows)
    table_regions: list[tuple[int, int]] = []  # (start_idx, end_idx)
    i = 0
    while i < len(row_info):
        if row_info[i]["is_table_row"]:
            start_idx = i
            while i < len(row_info) and row_info[i]["is_table_row"]:
                i += 1
            end_idx = i
            table_regions.append((start_idx, end_idx))
        else:
            i += 1

    # Check if enough rows are table rows (at least 20%)
    total_table_rows = sum(end - start for start, end in table_regions)
    if len(row_info) > 0 and total_table_rows / len(row_info) < 0.2:
        return None

    # Build output - collect table data first, then format with proper column widths
    result_lines: list[str] = []
    num_cols = len(global_columns)

    # Helper function to extract cells from a row
    def extract_cells(info: dict) -> list[str]:
        cells: list[str] = ["" for _ in range(num_cols)]
        for word in info["words"]:
            word_x = word["x0"]
            # Find the correct column using boundary ranges
            assigned_col = num_cols - 1  # Default to last column
            for col_idx in range(num_cols - 1):
                col_end = global_columns[col_idx + 1]
                if word_x < col_end - 20:
                    assigned_col = col_idx
                    break
            if cells[assigned_col]:
                cells[assigned_col] += " " + word["text"]
            else:
                cells[assigned_col] = word["text"]
        return cells

    # Process rows, collecting table data for proper formatting
    idx = 0
    while idx < len(row_info):
        info = row_info[idx]

        # Check if this row starts a table region
        table_region = None
        for start, end in table_regions:
            if idx == start:
                table_region = (start, end)
                break

        if table_region:
            start, end = table_region
            # Collect all rows in this table
            table_data: list[list[str]] = []
            for table_idx in range(start, end):
                cells = extract_cells(row_info[table_idx])
                table_data.append(cells)

            # Calculate column widths for this table
            if table_data:
                col_widths = [
                    max(len(row[col]) for row in table_data) for col in range(num_cols)
                ]
                # Ensure minimum width of 3 for separator dashes
                col_widths = [max(w, 3) for w in col_widths]

                # Format header row
                header = table_data[0]
                header_str = (
                    "| "
                    + " | ".join(
                        cell.ljust(col_widths[i]) for i, cell in enumerate(header)
                    )
                    + " |"
                )
                result_lines.append(header_str)

                # Format separator row
                separator = (
                    "| "
                    + " | ".join("-" * col_widths[i] for i in range(num_cols))
                    + " |"
                )
                result_lines.append(separator)

                # Format data rows
                for row in table_data[1:]:
                    row_str = (
                        "| "
                        + " | ".join(
                            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
                        )
                        + " |"
                    )
                    result_lines.append(row_str)

            idx = end  # Skip to end of table region
        else:
            # Check if we're inside a table region (not at start)
            in_table = False
            for start, end in table_regions:
                if start < idx < end:
                    in_table = True
                    break

            if not in_table:
                # Non-table content
                result_lines.append(info["text"])
            idx += 1

    return "\n".join(result_lines)


def _extract_tables_from_words(page: Any) -> list[list[list[str]]]:
    """
    Extract tables from a PDF page by analyzing word positions.
    This handles borderless tables where words are aligned in columns.

    This function is designed for structured tabular data (like invoices),
    not for multi-column text layouts in scientific documents.
    """
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return []

    # Group words by their Y position (rows)
    y_tolerance = 5
    rows_by_y: dict[float, list[dict]] = {}
    for word in words:
        y_key = round(word["top"] / y_tolerance) * y_tolerance
        if y_key not in rows_by_y:
            rows_by_y[y_key] = []
        rows_by_y[y_key].append(word)

    # Sort rows by Y position
    sorted_y_keys = sorted(rows_by_y.keys())

    # Find potential column boundaries by analyzing x positions across all rows
    all_x_positions = []
    for words_in_row in rows_by_y.values():
        for word in words_in_row:
            all_x_positions.append(word["x0"])

    if not all_x_positions:
        return []

    # Cluster x positions to find column starts
    all_x_positions.sort()
    x_tolerance_col = 20
    column_starts: list[float] = []
    for x in all_x_positions:
        if not column_starts or x - column_starts[-1] > x_tolerance_col:
            column_starts.append(x)

    # Need at least 3 columns but not too many (likely text layout, not table)
    if len(column_starts) < 3 or len(column_starts) > 10:
        return []

    # Find rows that span multiple columns (potential table rows)
    table_rows = []
    for y_key in sorted_y_keys:
        words_in_row = sorted(rows_by_y[y_key], key=lambda w: w["x0"])

        # Assign words to columns
        row_data = [""] * len(column_starts)
        for word in words_in_row:
            # Find the closest column
            best_col = 0
            min_dist = float("inf")
            for i, col_x in enumerate(column_starts):
                dist = abs(word["x0"] - col_x)
                if dist < min_dist:
                    min_dist = dist
                    best_col = i

            if row_data[best_col]:
                row_data[best_col] += " " + word["text"]
            else:
                row_data[best_col] = word["text"]

        # Only include rows that have content in multiple columns
        non_empty = sum(1 for cell in row_data if cell.strip())
        if non_empty >= 2:
            table_rows.append(row_data)

    # Validate table quality - tables should have:
    # 1. Enough rows (at least 3 including header)
    # 2. Short cell content (tables have concise data, not paragraphs)
    # 3. Consistent structure across rows
    if len(table_rows) < 3:
        return []

    # Check if cells contain short, structured data (not long text)
    long_cell_count = 0
    total_cell_count = 0
    for row in table_rows:
        for cell in row:
            if cell.strip():
                total_cell_count += 1
                # If cell has more than 30 chars, it's likely prose text
                if len(cell.strip()) > 30:
                    long_cell_count += 1

    # If more than 30% of cells are long, this is probably not a table
    if total_cell_count > 0 and long_cell_count / total_cell_count > 0.3:
        return []

    return [table_rows]


class PdfConverter(DocumentConverter):
    """
    Converts PDFs to Markdown.
    Supports extracting tables into aligned Markdown format (via pdfplumber).
    Falls back to pdfminer if pdfplumber is missing or fails.
    """

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".pdf",
                    feature="pdf",
                )
            ) from _dependency_exc_info[1].with_traceback(
                _dependency_exc_info[2]
            )  # type: ignore[union-attr]

        assert isinstance(file_stream, io.IOBase)

        markdown_chunks: list[str] = []

        # Read file stream into BytesIO for compatibility with pdfplumber
        pdf_bytes = io.BytesIO(file_stream.read())

        try:
            # Track how many pages are form-style vs plain text
            form_pages = 0
            plain_pages = 0

            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages:
                    # Try form-style word position extraction
                    page_content = _extract_form_content_from_words(page)

                    # If extraction returns None, this page is not form-style
                    if page_content is None:
                        plain_pages += 1
                        # Extract text using pdfplumber's basic extraction for this page
                        text = page.extract_text()
                        if text and text.strip():
                            markdown_chunks.append(text.strip())
                    else:
                        form_pages += 1
                        if page_content.strip():
                            markdown_chunks.append(page_content)

            # If most pages are plain text, use pdfminer for better text handling
            if plain_pages > form_pages and plain_pages > 0:
                pdf_bytes.seek(0)
                markdown = pdfminer.high_level.extract_text(pdf_bytes)
            else:
                # Build markdown from chunks
                markdown = "\n\n".join(markdown_chunks).strip()

        except Exception:
            # Fallback if pdfplumber fails
            pdf_bytes.seek(0)
            markdown = pdfminer.high_level.extract_text(pdf_bytes)

        # Fallback if still empty
        if not markdown:
            pdf_bytes.seek(0)
            markdown = pdfminer.high_level.extract_text(pdf_bytes)

        # Post-process to merge MasterFormat-style partial numbering with following text
        markdown = _merge_partial_numbering_lines(markdown)

        if kwargs.get("extract_pdf_images", False):
            image_links = _extract_pdf_images(
                pdf_bytes,
                stream_info=stream_info,
                pdf_image_dir=kwargs.get("pdf_image_dir"),
            )
            if image_links:
                images_section = ["## Extracted Images"]
                images_section.extend(
                    f"![PDF image {idx}]({path})"
                    for idx, path in enumerate(image_links, start=1)
                )
                markdown = (
                    markdown.rstrip() + "\n\n" + "\n".join(images_section)
                    if markdown
                    else "\n".join(images_section)
                )

        if kwargs.get("extract_pdf_figures", False):
            figure_links = _extract_pdf_figures(
                pdf_bytes,
                stream_info=stream_info,
                pdf_figure_dir=kwargs.get("pdf_figure_dir"),
                pdf_figure_dpi=kwargs.get("pdf_figure_dpi", 300),
            )
            if figure_links:
                figures_section = ["## Extracted Figures"]
                figures_section.extend(
                    f"![{caption}]({path})" for caption, path in figure_links
                )
                markdown = (
                    markdown.rstrip() + "\n\n" + "\n".join(figures_section)
                    if markdown
                    else "\n".join(figures_section)
                )

        return DocumentConverterResult(markdown=markdown)
