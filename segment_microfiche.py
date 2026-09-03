#!/usr/bin/env python3
"""
Microfiche page segmentation using Otsu thresholding.
Converts gigapixel JPG to 1-bit TIFF, finds page bounding boxes.
Supports both reading orders:
  - 'columns': left-to-right columns, top-to-bottom within each column
  - 'rows': top-to-bottom rows, left-to-right within each row
"""

import pyvips
import cv2
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import sys

# === CONFIGURATION ===
INPUT_FILE = "your_gigapixel_file.jpg"  # Change this or use --input
OUTPUT_DIR = Path("segmented")

# Reading order: 'columns' or 'rows'. Journals are always read like text -
# left-to-right, then down - so 'rows' is the default (2026-09-03). Page
# numbering is the downstream contract; a wrong order scrambles the journal
# silently.
READING_ORDER = 'rows'

# Skip header region (fraction of image height from top)
HEADER_SKIP_RATIO = 0.08  # Skip top 8% for yellow header

# Minimum page size (as fraction of image dimensions) to filter noise
MIN_PAGE_WIDTH_RATIO = 0.02
MIN_PAGE_HEIGHT_RATIO = 0.02

# Maximum grid size for validation
MAX_COLUMNS = 16
MAX_ROWS = 13

# Padding around detected pages (pixels) - added when extracting
PADDING = 0

# Erosion used to separate touching pages. Both passes shrink every blob by a
# known amount, which is added back to the boxes so they land on the true page
# edge rather than inside it.
DETECT_ERODE_KERNEL = 7      # global pass, at detect_scale
DETECT_ERODE_ITERATIONS = 2
REFINE_ERODE_KERNEL = 3      # local re-detection, at local_scale
REFINE_ERODE_ITERATIONS = 2

# Default crop margin, as a fraction of median page size. With the erosion bias
# compensated this is real safety headroom, not a correction. Generous on
# purpose (2026-09-03): real journals have unclear edges, and a sliver of card
# background in the crop is free while a sliver of lost text is not.
DEFAULT_PADDING_RATIO = 0.03


# Sentinel the OCR app's watcher polls: its presence means "fully written, safe
# to import". Everything about how it is written matters to that contract.
DONE_SENTINEL = "_done"

# Exit codes, so an app-driven run can tell failure modes apart
EXIT_NO_PAGES = 2


def prepare_card_dir(out_dir):
    """Clear a card folder so a re-run cannot be mistaken for a finished one.

    Removes the sentinel FIRST — while it exists the OCR app considers the card
    importable, so it must not survive into the rewrite — then empties pages/ so
    leftovers from a longer previous run cannot be imported as real pages.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentinel = out_dir / DONE_SENTINEL
    if sentinel.exists():
        sentinel.unlink()

    pages_dir = out_dir / "pages"
    if pages_dir.is_dir():
        for stale in pages_dir.iterdir():
            if stale.is_file():
                stale.unlink()


def write_done_sentinel(out_dir):
    """Publish the sentinel atomically, via temp file + rename."""
    out_dir = Path(out_dir)
    tmp = out_dir / f".{DONE_SENTINEL}.tmp"
    tmp.touch()
    os.replace(tmp, out_dir / DONE_SENTINEL)


# Panoramas/ is a work queue: a segmented card's panorama is moved out of it so
# what remains is what still needs doing. Sits alongside Panoramas/, not inside.
ARCHIVE_DIR_NAME = "PanoramaArchive"


def move_without_clobber(src, dest_dir):
    """Move a file into dest_dir, never overwriting something already there.

    Used both for archiving a finished panorama and for setting a failed scan
    aside. A collision means two different scans share a name, so the incoming
    one gets a numeric suffix rather than destroying the resident file.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1

    return Path(shutil.move(str(src), str(dest)))


# The masked header band is the card's only identifying metadata — title, part
# number, date, and the "N of M" card index. It is kept as page zero: it sorts
# ahead of page_001 (the OCR app orders on the last integer in the stem) and is
# trivial to drop downstream. Scaled right down; at 1/16 the text is still
# legible while the file is a fraction of a real page.
HEADER_PAGE_STEM = "page_000"
HEADER_PROXY_SCALE = 0.0625


# A page box may differ from the card's median page by this much before it is
# treated as something other than a page. Real cards run 97-98% size
# consistency, so this is deliberately loose — it exists to catch stitch
# artefacts (edge bands), not to police normal variation.
# A detection this many times wider/taller than the median page, while the
# other dimension is at or under the median, is a stitch band - not a page.
BAND_RATIO = 2.5


def drop_size_outliers(boxes, contours, band_ratio):
    """Reject band-shaped detections - never pages that are merely small.

    The minimum-size filter only catches specks. A bright band along a stitched
    card edge is the opposite problem: far too wide and too flat to be a page,
    but far too big to be filtered as noise. Left in, it becomes a blank page in
    the middle of the sequence and shifts every later page number by one.

    Shape, not size (2026-09-03): real journals hold pages of genuinely
    different sizes, so deviating from the median is not evidence against being
    a page. A band is unmistakable - grossly oversized in ONE dimension while
    at-or-under the median in the other. Everything else is kept: a wrongly
    kept blank costs one extra crop, a wrongly dropped page silently loses
    journal content.

    Returns (kept_boxes, kept_contours, dropped_boxes).
    """
    if len(boxes) < 4:
        # Too few to establish what "normal" looks like on this card.
        return list(boxes), contours, []

    median_w = np.median([b[2] for b in boxes])
    median_h = np.median([b[3] for b in boxes])
    if median_w <= 0 or median_h <= 0:
        return list(boxes), contours, []

    def is_page(box):
        _, _, w, h = box
        wide_band = w > median_w * band_ratio and h <= median_h
        tall_band = h > median_h * band_ratio and w <= median_w
        return not (wide_band or tall_band)

    keep = [i for i, b in enumerate(boxes) if is_page(b)]

    # If most boxes look wrong, the median itself is junk — trust nothing and
    # change nothing, rather than silently discarding most of the card.
    if len(keep) < len(boxes) / 2:
        return list(boxes), contours, []

    keep_set = set(keep)
    kept_boxes = [boxes[i] for i in keep]
    kept_contours = ([contours[i] for i in keep]
                     if contours is not None else contours)
    dropped = [b for i, b in enumerate(boxes) if i not in keep_set]
    return kept_boxes, kept_contours, dropped


def erosion_radius(kernel_size, iterations):
    """Pixels eaten from each side of a blob by cv2.erode with this kernel."""
    return (kernel_size // 2) * iterations


def expand_boxes(boxes, radius, max_width, max_height):
    """Grow each (x, y, w, h) box by radius per side, clamped to image bounds."""
    if radius == 0:
        return list(boxes)

    expanded = []
    for x, y, w, h in boxes:
        left = max(0, x - radius)
        top = max(0, y - radius)
        right = min(max_width, x + w + radius)
        bottom = min(max_height, y + h + radius)
        expanded.append((left, top, right - left, bottom - top))
    return expanded


def compute_otsu_threshold(gray_image, sample_scale=0.01):
    """Compute Otsu threshold from a thumbnail sample."""
    thumb = gray_image.resize(sample_scale)
    thumb_np = np.ndarray(
        buffer=thumb.write_to_memory(),
        dtype=np.uint8,
        shape=[thumb.height, thumb.width]
    )
    thresh, _ = cv2.threshold(thumb_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def sort_boxes_by_columns(boxes, tolerance_ratio=0.5):
    """Sort boxes: left-to-right by column, top-to-bottom within each column."""
    if not boxes:
        return []

    # Sort by x first
    boxes_sorted = sorted(boxes, key=lambda b: b[0])

    # Group into columns based on x overlap
    avg_width = np.mean([b[2] for b in boxes])
    tolerance = avg_width * tolerance_ratio

    columns = []
    current_column = [boxes_sorted[0]]

    for box in boxes_sorted[1:]:
        # Check if this box is in the same column as previous
        prev_x = current_column[-1][0]
        if abs(box[0] - prev_x) < tolerance:
            current_column.append(box)
        else:
            # New column
            columns.append(sorted(current_column, key=lambda b: b[1]))  # Sort by y
            current_column = [box]

    columns.append(sorted(current_column, key=lambda b: b[1]))

    # Flatten
    result = []
    for col in columns:
        result.extend(col)

    return result


def sort_boxes_by_rows(boxes, tolerance_ratio=0.5):
    """Sort boxes: top-to-bottom by row, left-to-right within each row."""
    if not boxes:
        return []

    # Sort by y first
    boxes_sorted = sorted(boxes, key=lambda b: b[1])

    # Group into rows based on y overlap
    avg_height = np.mean([b[3] for b in boxes])
    tolerance = avg_height * tolerance_ratio

    rows = []
    current_row = [boxes_sorted[0]]

    for box in boxes_sorted[1:]:
        # Check if this box is in the same row as previous
        prev_y = current_row[-1][1]
        if abs(box[1] - prev_y) < tolerance:
            current_row.append(box)
        else:
            # New row
            rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort by x
            current_row = [box]

    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Flatten
    result = []
    for row in rows:
        result.extend(row)

    return result


def compute_card_quality(boxes, contours):
    """Compute card-level quality score (0-100) for segmentation results.

    Components:
    - Size consistency (30%): How uniform page sizes are
    - Grid alignment (40%): How well pages align to a grid
    - Spacing regularity (20%): How uniform gaps between pages are
    - Shape regularity (10%): How rectangular the detections are
    """
    if len(boxes) < 2:
        # 'grid' included even here: callers print it unconditionally.
        return {'total': 100.0, 'size': 100.0, 'alignment': 100.0,
                'spacing': 100.0, 'shape': 100.0,
                'grid': f"{len(boxes)}x{len(boxes)}"}

    widths = np.array([b[2] for b in boxes])
    heights = np.array([b[3] for b in boxes])
    xs = np.array([b[0] for b in boxes])
    ys = np.array([b[1] for b in boxes])

    # --- 1. Size consistency (30%) ---
    w_median = np.median(widths)
    h_median = np.median(heights)
    w_cv = np.std(widths) / w_median if w_median > 0 else 0
    h_cv = np.std(heights) / h_median if h_median > 0 else 0
    size_score = max(0.0, 100.0 * (1.0 - (w_cv + h_cv) * 2.0))

    # --- 2. Grid alignment (40%) ---
    avg_width = np.mean(widths)
    col_tolerance = avg_width * 0.5

    sorted_by_x = sorted(range(len(boxes)), key=lambda i: xs[i])
    columns = [[sorted_by_x[0]]]
    for idx in sorted_by_x[1:]:
        if abs(xs[idx] - xs[columns[-1][-1]]) < col_tolerance:
            columns[-1].append(idx)
        else:
            columns.append([idx])

    avg_height = np.mean(heights)
    row_tolerance = avg_height * 0.5

    sorted_by_y = sorted(range(len(boxes)), key=lambda i: ys[i])
    rows = [[sorted_by_y[0]]]
    for idx in sorted_by_y[1:]:
        if abs(ys[idx] - ys[rows[-1][-1]]) < row_tolerance:
            rows[-1].append(idx)
        else:
            rows.append([idx])

    col_spreads = []
    for col in columns:
        if len(col) > 1:
            spread = np.std(xs[col]) / avg_width if avg_width > 0 else 0
            col_spreads.append(spread)

    row_spreads = []
    for row in rows:
        if len(row) > 1:
            spread = np.std(ys[row]) / avg_height if avg_height > 0 else 0
            row_spreads.append(spread)

    avg_col_spread = np.mean(col_spreads) if col_spreads else 0
    avg_row_spread = np.mean(row_spreads) if row_spreads else 0
    alignment_score = max(0.0, 100.0 * (1.0 - (avg_col_spread + avg_row_spread) * 5.0))

    # --- 3. Spacing regularity (20%) ---
    col_centers = sorted(np.mean(xs[col]) for col in columns)
    col_gaps = np.diff(col_centers) if len(col_centers) > 1 else np.array([])

    row_centers = sorted(np.mean(ys[row]) for row in rows)
    row_gaps = np.diff(row_centers) if len(row_centers) > 1 else np.array([])

    gap_scores = []
    if len(col_gaps) > 1:
        col_gap_cv = np.std(col_gaps) / np.mean(col_gaps) if np.mean(col_gaps) > 0 else 0
        gap_scores.append(max(0.0, 100.0 * (1.0 - col_gap_cv * 3.0)))
    else:
        gap_scores.append(100.0)
    if len(row_gaps) > 1:
        row_gap_cv = np.std(row_gaps) / np.mean(row_gaps) if np.mean(row_gaps) > 0 else 0
        gap_scores.append(max(0.0, 100.0 * (1.0 - row_gap_cv * 3.0)))
    else:
        gap_scores.append(100.0)
    spacing_score = float(np.mean(gap_scores))

    # --- 4. Shape regularity (10%) ---
    if contours is not None and len(contours) > 0:
        rects = []
        for c in contours:
            area = cv2.contourArea(c)
            _, _, cw, ch = cv2.boundingRect(c)
            rect_area = cw * ch
            if rect_area > 0:
                rects.append(area / rect_area)
        shape_score = float(np.mean(rects)) * 100.0 if rects else 100.0
    else:
        shape_score = 100.0

    total = (size_score * 0.30 + alignment_score * 0.40 +
             spacing_score * 0.20 + shape_score * 0.10)

    return {
        'total': round(total, 1),
        'size': round(size_score, 1),
        'alignment': round(alignment_score, 1),
        'spacing': round(spacing_score, 1),
        'shape': round(shape_score, 1),
        'grid': f"{len(columns)}x{len(rows)}",
    }


def refine_box_local(input_file, box, otsu_thresh, orig_w, orig_h,
                     invert=False, header_skip_px=0):
    """Refine a bounding box by re-detecting the page at higher local resolution.

    Extracts a padded region around the approximate box (capturing edges of
    neighboring pages), applies threshold + contour detection at ~20% of full-res,
    and picks the contour closest to center as the precise page boundary.
    """
    x, y, w, h = box

    # Padding to include edges of neighboring pages
    pad_x = int(w * 0.45)
    pad_y = int(h * 0.45)

    rx = max(0, x - pad_x)
    ry = max(0, y - pad_y)
    rw = min(w + 2 * pad_x, orig_w - rx)
    rh = min(h + 2 * pad_y, orig_h - ry)

    # Load region (pyvips reads only the needed tiles)
    img = pyvips.Image.new_from_file(input_file, access='random')
    region = img.crop(rx, ry, rw, rh)
    if region.bands > 1:
        region = region.colourspace('b-w')

    # Downsample locally — 20% of full-res gives ~660×500 per region, fast for OpenCV
    local_scale = 0.2
    region_small = region.resize(local_scale)

    region_np = np.ndarray(
        buffer=region_small.write_to_memory(),
        dtype=np.uint8,
        shape=[region_small.height, region_small.width]
    )

    # Threshold (same Otsu value as global pass)
    binary = (region_np >= otsu_thresh).astype(np.uint8) * 255
    if invert:
        binary = cv2.bitwise_not(binary)

    # Mask out header region (same as global pass)
    if header_skip_px > 0 and ry < header_skip_px:
        mask_rows = int((header_skip_px - ry) * local_scale)
        if mask_rows > 0:
            binary[:mask_rows, :] = 0

    # Focus mask: black out everything beyond ~110% of expected page size
    # This prevents bright empty-neighbor areas from merging with the page
    # 5% per side ≈ half the typical inter-page gap
    margin = 1.1
    exp_left = int((x - rx - w * (margin - 1) / 2) * local_scale)
    exp_top = int((y - ry - h * (margin - 1) / 2) * local_scale)
    exp_right = int(exp_left + w * margin * local_scale)
    exp_bottom = int(exp_top + h * margin * local_scale)
    exp_left = max(0, exp_left)
    exp_top = max(0, exp_top)
    exp_right = min(binary.shape[1], exp_right)
    exp_bottom = min(binary.shape[0], exp_bottom)
    mask = np.zeros_like(binary)
    mask[exp_top:exp_bottom, exp_left:exp_right] = 255
    binary = cv2.bitwise_and(binary, mask)

    # Morphological close to fill text/image holes within pages
    close_kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Light erosion to ensure neighboring pages stay separated
    erode_kernel = np.ones((REFINE_ERODE_KERNEL, REFINE_ERODE_KERNEL), np.uint8)
    binary = cv2.erode(binary, erode_kernel, iterations=REFINE_ERODE_ITERATIONS)

    contours_local, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick contour closest to center that's large enough to be a page
    center_x = rw * local_scale / 2
    center_y = rh * local_scale / 2
    min_area = w * h * local_scale * local_scale * 0.2

    best = None
    best_dist = float('inf')

    for c in contours_local:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        dist = abs(bx + bw / 2 - center_x) + abs(by + bh / 2 - center_y)
        if dist < best_dist:
            best_dist = dist
            best = (bx, by, bw, bh)

    if best is None:
        return box  # fallback

    # Undo this pass's erosion shrink before scaling back, matching the global pass
    refine_radius = erosion_radius(REFINE_ERODE_KERNEL, REFINE_ERODE_ITERATIONS)
    (bx, by, bw, bh) = expand_boxes(
        [best], refine_radius, binary.shape[1], binary.shape[0])[0]

    sb = 1.0 / local_scale
    refined = (
        rx + int(bx * sb),
        ry + int(by * sb),
        int(bw * sb),
        int(bh * sb),
    )

    # Sanity check: reject if center shifted >20% or size changed >20%
    ref_cx = refined[0] + refined[2] / 2
    ref_cy = refined[1] + refined[3] / 2
    orig_cx = x + w / 2
    orig_cy = y + h / 2
    if (abs(ref_cx - orig_cx) > w * 0.2 or abs(ref_cy - orig_cy) > h * 0.2
            or refined[2] < w * 0.8 or refined[2] > w * 1.2
            or refined[3] < h * 0.8 or refined[3] > h * 1.2):
        return box  # fallback

    return refined


def main():
    parser = argparse.ArgumentParser(description='Segment microfiche pages')
    parser.add_argument('--input', '-i', default=INPUT_FILE, help='Input image file')
    parser.add_argument('--order', '-o', choices=['columns', 'rows'], default=READING_ORDER,
                        help='Reading order: columns (down then right) or rows (right then down)')
    parser.add_argument('--header-skip', '-hs', type=float, default=HEADER_SKIP_RATIO,
                        help='Fraction of image height to skip at top (for header)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert binary image (if pages are dark on light background)')
    parser.add_argument('--padding', '-p', type=float, default=None,
                        help='Padding around detected pages (pixels, or percent if 0-1). Default: 1%%')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Only output coordinates, do not extract pages')
    parser.add_argument('--format', choices=['tif', 'jpg'], default='tif',
                        help='Page crop format. Default tif (LZW): the decided '
                             'workflow is TIFF end-to-end until final packaging '
                             '— a JPG crop here would re-encode already-stitched '
                             'pixels and the archival TIFF downstream would '
                             'inherit the artifacts.')
    parser.add_argument('--refine', action='store_true',
                        help='Refine page positions using local high-res re-detection')
    parser.add_argument('--output', '-O', default=None,
                        help='Output directory (default: segmented/<card_name>/ next to input)')
    parser.add_argument('--no-archive', action='store_true',
                        help='Leave the source panorama in place. By default a '
                             'successfully segmented panorama is moved to '
                             f'../{ARCHIVE_DIR_NAME}/ so Panoramas/ holds only '
                             'what still needs doing.')
    parser.add_argument('--archive-dir', default=None,
                        help=f'Where to archive the panorama (default: '
                             f'{ARCHIVE_DIR_NAME}/ beside the input folder)')
    parser.add_argument('--no-header-page', action='store_true',
                        help='Do not write the masked header band as '
                             'pages/page_000.tif (contract C11).')
    parser.add_argument('--header-page', action='store_true',
                        help='Deprecated no-op: writing the header is the default '
                             'as of 2026-08-23. Still accepted so existing callers '
                             'do not fail — argparse exits 2 on an unknown flag, '
                             'which collides with EXIT_NO_PAGES.')
    parser.add_argument('--debug', action='store_true',
                        help='Also write the binary TIFF and box overlay to <card>/_debug/. '
                             'Off by default: the OCR app reads loose image files in the '
                             'card folder as pages.')
    args = parser.parse_args()

    input_file = args.input
    input_path = Path(input_file)

    # Output directory: next to input file, per-card subfolder
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = input_path.parent / "segmented" / input_path.stem

    if args.skip_extraction:
        # Inspection mode writes coordinates only. Clearing here would delete a
        # finished card's pages and retract a sentinel that is still accurate.
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Clear any previous run before writing: a stale _done would tell the
        # OCR app this card is importable while we are still rewriting it.
        prepare_card_dir(out_dir)

    pages_dir = out_dir / "pages"
    csv_path = out_dir / "page_coordinates.csv"

    # The visualization is the operator's ground truth for what was detected,
    # in which order - written on EVERY run (2026-09-03), inside _debug/ so the
    # OCR app never mistakes it for a page. Heavy artifacts (the full-res
    # binary) still hide behind --debug.
    debug_dir = out_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    temp_tiff = debug_dir / "temp_binary.tif"
    viz_path = debug_dir / "visualization.jpg"

    # === STEP 1: Load image and convert to binary TIFF ===
    print(f"Loading {input_file} with libvips...")
    image = pyvips.Image.new_from_file(input_file, access='random')

    original_width = image.width
    original_height = image.height
    print(f"Image size: {original_width} x {original_height}")

    # Convert to grayscale if needed
    if image.bands > 1:
        gray = image.colourspace('b-w')
    else:
        gray = image

    # Compute Otsu threshold from thumbnail
    print("Computing Otsu threshold from thumbnail...")
    otsu_thresh = compute_otsu_threshold(gray)
    print(f"Otsu threshold: {otsu_thresh}")

    # Apply threshold to full image
    print("Applying threshold to full image...")
    binary = gray >= otsu_thresh

    # Downsample for contour detection (OpenCV has pixel limits)
    # Use 10% scale for detection, then scale coordinates back
    detect_scale = 0.1
    print(f"Downsampling to {detect_scale*100:.0f}% for contour detection...")
    binary_small = binary.resize(detect_scale)

    # Convert to numpy for OpenCV
    binary_img = np.ndarray(
        buffer=binary_small.write_to_memory(),
        dtype=np.uint8,
        shape=[binary_small.height, binary_small.width]
    )
    # Convert boolean (0/1) to grayscale (0/255)
    binary_img = (binary_img * 255).astype(np.uint8)

    # Save full-res binary TIFF for reference (optional)
    if args.debug:
        print(f"Saving 1-bit TIFF to {temp_tiff}...")
        binary.write_to_file(str(temp_tiff), compression='lzw', bigtiff=True)

    # Free memory
    del image, gray, binary, binary_small

    # === STEP 2: Find contours in the binary image ===
    print("Finding contours on downsampled image...")

    # Invert if needed
    if args.invert:
        print("Inverting binary image...")
        binary_img = cv2.bitwise_not(binary_img)

    # Calculate header skip in pixels (on downsampled image)
    header_skip_px_small = int(original_height * detect_scale * args.header_skip)
    print(f"Skipping top {header_skip_px_small} pixels in downsampled image (header region)")

    # Mask out header region
    if header_skip_px_small > 0:
        binary_img[:header_skip_px_small, :] = 0

    # Apply erosion to separate touching pages
    # Kernel size depends on gap between pages (at 10% scale, ~5-10 pixels)
    print(f"Applying erosion (kernel={DETECT_ERODE_KERNEL}) to separate pages...")
    kernel = np.ones((DETECT_ERODE_KERNEL, DETECT_ERODE_KERNEL), np.uint8)
    binary_img = cv2.erode(binary_img, kernel, iterations=DETECT_ERODE_ITERATIONS)

    print("Finding contours...")
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === STEP 3: Filter and sort bounding boxes ===
    # Min sizes on downsampled image
    min_w = int(original_width * detect_scale * MIN_PAGE_WIDTH_RATIO)
    min_h = int(original_height * detect_scale * MIN_PAGE_HEIGHT_RATIO)

    boxes = []
    filtered_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= min_h:
            boxes.append((x, y, w, h))
            filtered_contours.append(c)

    print(f"Found {len(boxes)} potential pages")

    # Fail loudly on a total detection failure. Writing an empty card folder
    # would be worse than useless: the OCR app skips empty folders silently, so
    # the card would vanish from the queue with no error anywhere.
    if not boxes:
        print(f"\nERROR: no pages detected in {input_file}", file=sys.stderr)
        print("  No _done sentinel written — this card will not be offered for import.",
              file=sys.stderr)
        # A no-pages failure is exactly when the picture matters most: write
        # the thresholded view so the operator can SEE what detection saw.
        fail_scale = min(1.0, 2000 / max(binary_img.shape[1], binary_img.shape[0]))
        fail_viz = cv2.resize(binary_img, None, fx=fail_scale, fy=fail_scale)
        cv2.imwrite(str(viz_path), fail_viz)
        print(f"  Detection view saved to {viz_path}", file=sys.stderr)
        if not args.skip_extraction:
            moved = move_without_clobber(input_path, input_path.parent / "error")
            print(f"  Source scan moved to {moved}", file=sys.stderr)
            # The card folder stays: it holds _debug/visualization.jpg and no
            # sentinel, so the OCR app skips it while a human can inspect it.
            try:
                out_dir.rmdir()
            except OSError:
                pass  # expected - _debug/ is in there
        else:
            print("  Source left in place (--skip-extraction is inspection-only).",
                  file=sys.stderr)
        return EXIT_NO_PAGES

    # Undo the erosion shrink so boxes sit on the true page edge. Erosion of a
    # rectangle removes exactly this many pixels per side, so the recovery is
    # exact rather than a fudge factor.
    detect_radius = erosion_radius(DETECT_ERODE_KERNEL, DETECT_ERODE_ITERATIONS)
    boxes = expand_boxes(boxes, detect_radius, binary_img.shape[1], binary_img.shape[0])
    print(f"Compensating erosion: +{detect_radius}px per side "
          f"(+{int(detect_radius / detect_scale)}px at full resolution)")

    # Reject anything that is not page-shaped for this card (stitch edge bands)
    boxes, filtered_contours, dropped = drop_size_outliers(
        boxes, filtered_contours, BAND_RATIO)
    if dropped:
        scale_up = 1.0 / detect_scale
        print(f"Rejected {len(dropped)} detection(s) unlike this card's median page:")
        for (x, y, w, h) in dropped:
            print(f"  at ({int(x * scale_up)}, {int(y * scale_up)}) "
                  f"size {int(w * scale_up)} x {int(h * scale_up)} full-res")
        print(f"{len(boxes)} pages remain")

    # === Compute card quality score ===
    quality = compute_card_quality(boxes, filtered_contours)

    # Validate against max grid size
    max_pages = MAX_COLUMNS * MAX_ROWS
    if len(boxes) > max_pages:
        print(f"Warning: Found {len(boxes)} pages, exceeds max grid {MAX_COLUMNS}x{MAX_ROWS}={max_pages}")
        print("Consider adjusting MIN_PAGE_WIDTH_RATIO or MIN_PAGE_HEIGHT_RATIO")

    # Sort based on reading order
    if args.order == 'columns':
        print("Sorting by columns (left-to-right, then top-to-bottom within each column)")
        boxes_sorted = sort_boxes_by_columns(boxes)
    else:
        print("Sorting by rows (top-to-bottom, then left-to-right within each row)")
        boxes_sorted = sort_boxes_by_rows(boxes)

    # === STEP 4: Scale coordinates back to original size ===
    scale_back = 1.0 / detect_scale
    boxes_fullres = []
    for (x, y, w, h) in boxes_sorted:
        boxes_fullres.append((
            int(x * scale_back),
            int(y * scale_back),
            int(w * scale_back),
            int(h * scale_back)
        ))

    # === STEP 4b: Optionally refine positions at higher local resolution ===
    if args.refine:
        print("\n=== REFINING PAGE POSITIONS ===")
        print("Re-detecting each page locally at 20% resolution...")

        header_skip_fullres = int(original_height * args.header_skip)

        def _refine_one(box):
            return refine_box_local(
                input_file, box, otsu_thresh,
                original_width, original_height,
                invert=args.invert, header_skip_px=header_skip_fullres)

        refined = [None] * len(boxes_fullres)
        refine_done = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_refine_one, b): i
                       for i, b in enumerate(boxes_fullres)}
            for future in as_completed(futures):
                idx = futures[future]
                refined[idx] = future.result()
                refine_done += 1
                if refine_done % 20 == 0 or refine_done == len(boxes_fullres):
                    print(f"  Progress: {refine_done}/{len(boxes_fullres)}")

        # Report how much positions shifted
        shifts_x = [abs(r[0] - o[0]) for r, o in zip(refined, boxes_fullres)]
        shifts_y = [abs(r[1] - o[1]) for r, o in zip(refined, boxes_fullres)]
        print(f"  Avg shift: {np.mean(shifts_x):.0f}px x, {np.mean(shifts_y):.0f}px y")
        print(f"  Max shift: {max(shifts_x):.0f}px x, {max(shifts_y):.0f}px y")

        boxes_fullres = refined

    # Output coordinates
    print("\n=== PAGE COORDINATES (full resolution) ===")
    print("Page#, X, Y, Width, Height")
    for i, (x, y, w, h) in enumerate(boxes_fullres, 1):
        print(f"{i:3d}, {x}, {y}, {w}, {h}")

    # Print card quality score
    q = quality['total']
    grade = "GOOD" if q > 80 else ("FAIR" if q >= 60 else "POOR")
    print(f"\n{'=' * 42}")
    print(f"  Card Quality: {q}/100  ({grade})")
    print(f"  Detected grid: {quality['grid']}")
    print(f"{'=' * 42}")
    print(f"  Size consistency .. {quality['size']:5.1f}  (30%)")
    print(f"  Grid alignment ... {quality['alignment']:5.1f}  (40%)")
    print(f"  Spacing regularity {quality['spacing']:5.1f}  (20%)")
    print(f"  Shape regularity . {quality['shape']:5.1f}  (10%)")
    print(f"{'=' * 42}")

    # Save to CSV
    with open(csv_path, 'w') as f:
        f.write(f"# Card Quality: {quality['total']}/100 ({grade})"
                f" | grid={quality['grid']}"
                f" | size={quality['size']}"
                f" | align={quality['alignment']}"
                f" | spacing={quality['spacing']}"
                f" | shape={quality['shape']}\n")
        f.write("page,x,y,width,height\n")
        for i, (x, y, w, h) in enumerate(boxes_fullres, 1):
            f.write(f"{i},{x},{y},{w},{h}\n")
    print(f"\nCoordinates saved to {csv_path}")

    # === STEP 5: Create visualization ===
    print("Creating visualization...")
    # binary_img is already downsampled, resize further if needed
    detect_height, detect_width = binary_img.shape[:2]
    viz_scale = min(1.0, 2000 / max(detect_width, detect_height))
    viz = cv2.resize(binary_img, None, fx=viz_scale, fy=viz_scale)
    viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    # Use full-res boxes (possibly refined) scaled down to viz coordinates
    fullres_to_viz = detect_scale * viz_scale
    for i, (x, y, w, h) in enumerate(boxes_fullres, 1):
        sx, sy = int(x * fullres_to_viz), int(y * fullres_to_viz)
        sw, sh = int(w * fullres_to_viz), int(h * fullres_to_viz)
        cv2.rectangle(viz, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(viz, str(i), (sx + 5, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add quality score banner at the top
    q = quality['total']
    if q > 80:
        banner_color = (0, 180, 0)     # green
    elif q >= 60:
        banner_color = (0, 200, 220)   # yellow (BGR)
    else:
        banner_color = (0, 0, 200)     # red
    banner_h = 32
    banner = np.zeros((banner_h, viz.shape[1], 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    label = f"Card Quality: {q}/100 ({grade})  |  {quality['grid']}  |  size={quality['size']}  align={quality['alignment']}  spacing={quality['spacing']}  shape={quality['shape']}"
    cv2.putText(banner, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2)
    viz = np.vstack([banner, viz])

    cv2.imwrite(str(viz_path), viz)
    print(f"Saved {viz_path}")

    # === STEP 6: Extract pages (optional) ===
    if not args.skip_extraction:
        print("\n=== EXTRACTING PAGES (parallel) ===")
        pages_dir.mkdir(exist_ok=True)

        # Number of parallel workers
        num_workers = 5
        print(f"Using {num_workers} parallel workers")

        # Compute padding: default 1% of median page size (hairline safety margin)
        if args.padding is None:
            median_w = int(np.median([b[2] for b in boxes_fullres]))
            median_h = int(np.median([b[3] for b in boxes_fullres]))
            pad_x = int(median_w * DEFAULT_PADDING_RATIO)
            pad_y = int(median_h * DEFAULT_PADDING_RATIO)
        elif args.padding <= 1.0:
            median_w = int(np.median([b[2] for b in boxes_fullres]))
            median_h = int(np.median([b[3] for b in boxes_fullres]))
            pad_x = int(median_w * args.padding)
            pad_y = int(median_h * args.padding)
        else:
            pad_x = pad_y = int(args.padding)
        print(f"Crop margin: {pad_x}px x {pad_y}px")

        def extract_page(task):
            """Extract a single page from the source image."""
            i, x, y, w, h, src_file, out_dir, orig_w, orig_h = task

            img = pyvips.Image.new_from_file(src_file, access='random')

            # Apply margin
            px = max(0, x - pad_x)
            py = max(0, y - pad_y)
            pw = min(w + 2 * pad_x, orig_w - px)
            ph = min(h + 2 * pad_y, orig_h - py)

            page = img.crop(px, py, pw, ph)

            if args.format == 'jpg':
                output_path = out_dir / f"page_{i:03d}.jpg"
                page.write_to_file(str(output_path), Q=95)
            else:
                output_path = out_dir / f"page_{i:03d}.tif"
                page.write_to_file(str(output_path), compression='lzw')
            return i, output_path

        # Prepare tasks
        tasks = [
            (i, x, y, w, h, input_file, pages_dir, original_width, original_height)
            for i, (x, y, w, h) in enumerate(boxes_fullres, 1)
        ]

        # Execute in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_page, task): task[0] for task in tasks}
            for future in as_completed(futures):
                i, path = future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"  Progress: {completed}/{len(tasks)} pages extracted")

        print(f"\nExtracted {len(boxes_fullres)} pages to {pages_dir}/")

        # Keep the header band as page zero. Everything above the first page row
        # is masked during detection, and it carries the only text identifying
        # the card, so throwing it away loses the card's identity.
        header_px = int(original_height * args.header_skip)
        if not args.no_header_page and header_px > 0:
            src = pyvips.Image.new_from_file(input_file, access='random')
            header = src.crop(0, 0, original_width, header_px).resize(HEADER_PROXY_SCALE)
            header_path = pages_dir / f"{HEADER_PAGE_STEM}.tif"
            header.write_to_file(str(header_path), compression='lzw')
            print(f"Header kept as {header_path.name} "
                  f"({header.width} x {header.height}, "
                  f"{HEADER_PROXY_SCALE:.4g} scale of the masked band)")

        # Sentinel written LAST, after every page is on disk, and atomically:
        # the OCR app's watch+confirm list treats its presence as "fully
        # written" and only then offers the card for import (without it the app
        # falls back to a 120 s quiet-period heuristic).
        write_done_sentinel(out_dir)

        # Card is complete and published; take the panorama out of the queue.
        # After the sentinel, never before: if this move fails the card is still
        # valid and importable, it just needs filing by hand.
        if not args.no_archive:
            archive_dir = (Path(args.archive_dir) if args.archive_dir
                           else input_path.parent.parent / ARCHIVE_DIR_NAME)
            try:
                archived = move_without_clobber(input_path, archive_dir)
                print(f"Panorama archived to {archived}")
            except OSError as exc:
                print(f"WARNING: could not archive {input_path}: {exc}",
                      file=sys.stderr)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
