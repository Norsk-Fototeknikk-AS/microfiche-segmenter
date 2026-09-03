"""Tests for page-box geometry.

The detection pass erodes the binary image to separate touching pages, which
shrinks every blob by a known amount. These tests pin down that the shrink is
compensated exactly, so boxes land on the true page edge.
"""

import cv2
import numpy as np
from pathlib import Path

from segment_microfiche import (
    DETECT_ERODE_ITERATIONS,
    DETECT_ERODE_KERNEL,
    erosion_radius,
    expand_boxes,
)


def test_erosion_radius_matches_kernel_and_iterations():
    # A k x k kernel eats k//2 pixels per side per iteration.
    assert erosion_radius(7, 2) == 6
    assert erosion_radius(3, 2) == 2
    assert erosion_radius(5, 1) == 2


def test_expand_recovers_a_rectangle_shrunk_by_the_detect_erosion():
    """The real pipeline, in miniature: erode, boundingRect, expand back."""
    img = np.zeros((400, 400), np.uint8)
    truth = (100, 120, 180, 150)  # x, y, w, h
    x, y, w, h = truth
    img[y:y + h, x:x + w] = 255

    kernel = np.ones((DETECT_ERODE_KERNEL, DETECT_ERODE_KERNEL), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=DETECT_ERODE_ITERATIONS)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shrunk = [cv2.boundingRect(c) for c in contours]

    radius = erosion_radius(DETECT_ERODE_KERNEL, DETECT_ERODE_ITERATIONS)
    recovered = expand_boxes(shrunk, radius, img.shape[1], img.shape[0])

    assert len(recovered) == 1
    for got, want in zip(recovered[0], truth):
        assert abs(got - want) <= 1


def test_expand_without_compensation_undershoots():
    """Guards the premise: skipping the expansion really does lose the edge."""
    img = np.zeros((400, 400), np.uint8)
    img[120:270, 100:280] = 255

    kernel = np.ones((DETECT_ERODE_KERNEL, DETECT_ERODE_KERNEL), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=DETECT_ERODE_ITERATIONS)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    radius = erosion_radius(DETECT_ERODE_KERNEL, DETECT_ERODE_ITERATIONS)
    assert x == 100 + radius
    assert w == 180 - 2 * radius


def test_expand_clamps_at_image_bounds():
    boxes = [(2, 3, 50, 50)]
    (x, y, w, h) = expand_boxes(boxes, 6, 100, 100)[0]
    assert (x, y) == (0, 0)
    # Left edge clamped at 0, so width grows only by what was available there.
    assert x + w == 2 + 50 + 6
    assert y + h == 3 + 50 + 6


def test_expand_clamps_at_far_edge():
    boxes = [(40, 40, 55, 55)]
    (x, y, w, h) = expand_boxes(boxes, 10, 100, 100)[0]
    assert x == 30 and y == 30
    assert x + w == 100
    assert y + h == 100


def test_expand_with_zero_radius_is_identity():
    boxes = [(10, 20, 30, 40), (50, 60, 70, 80)]
    assert expand_boxes(boxes, 0, 1000, 1000) == boxes


# --- Card-folder lifecycle -------------------------------------------------
# The OCR app treats _done as "safe to import". These tests pin the invariants
# its watcher depends on: no stale _done during a rewrite, no stale pages
# surviving a re-run, and _done appearing atomically.

from segment_microfiche import (
    DONE_SENTINEL,
    move_without_clobber,
    prepare_card_dir,
    write_done_sentinel,
)


def test_prepare_removes_stale_done_before_a_rerun(tmp_path):
    (tmp_path / DONE_SENTINEL).touch()
    prepare_card_dir(tmp_path)
    assert not (tmp_path / DONE_SENTINEL).exists()


def test_prepare_clears_stale_pages(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    for n in (1, 2, 147):
        (pages / f"page_{n:03d}.tif").write_bytes(b"old")
    prepare_card_dir(tmp_path)
    assert pages.exists()
    assert list(pages.iterdir()) == []


def test_prepare_is_safe_on_a_fresh_folder(tmp_path):
    fresh = tmp_path / "612130000333_0000736115"
    prepare_card_dir(fresh)
    assert fresh.is_dir()
    assert not (fresh / DONE_SENTINEL).exists()


def test_prepare_leaves_unrelated_files_alone(tmp_path):
    (tmp_path / "header.json").write_text("{}")
    prepare_card_dir(tmp_path)
    assert (tmp_path / "header.json").exists()


def test_write_done_sentinel_creates_the_marker(tmp_path):
    write_done_sentinel(tmp_path)
    assert (tmp_path / DONE_SENTINEL).is_file()


def test_write_done_sentinel_leaves_no_temp_file_behind(tmp_path):
    write_done_sentinel(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [DONE_SENTINEL]


def test_write_done_sentinel_is_repeatable(tmp_path):
    write_done_sentinel(tmp_path)
    write_done_sentinel(tmp_path)
    assert (tmp_path / DONE_SENTINEL).is_file()


def test_error_move_relocates_the_source(tmp_path):
    src = tmp_path / "612130000333_0000736115 Panorama.jpg"
    src.write_bytes(b"scan")
    dest = move_without_clobber(src, tmp_path / "error")
    assert not src.exists()
    assert dest.read_bytes() == b"scan"
    assert dest.parent.name == "error"


def test_error_move_does_not_clobber_an_earlier_failure(tmp_path):
    err = tmp_path / "error"
    err.mkdir()
    (err / "card.jpg").write_bytes(b"first failure")

    src = tmp_path / "card.jpg"
    src.write_bytes(b"second failure")
    dest = move_without_clobber(src, err)

    assert (err / "card.jpg").read_bytes() == b"first failure"
    assert dest.read_bytes() == b"second failure"
    assert dest.name != "card.jpg"


# --- Quality score, degenerate cards ---------------------------------------

from segment_microfiche import compute_card_quality


def test_quality_reports_a_grid_for_a_single_box():
    """main() reads quality['grid'] unconditionally, so it must always exist."""
    q = compute_card_quality([(0, 0, 100, 100)], None)
    assert q['grid'] == "1x1"


def test_quality_reports_a_grid_for_no_boxes():
    q = compute_card_quality([], None)
    assert q['grid'] == "0x0"


def test_quality_always_carries_the_keys_main_prints():
    for boxes in ([], [(0, 0, 10, 10)], [(0, 0, 10, 10), (20, 0, 10, 10)]):
        q = compute_card_quality(boxes, None)
        assert set(q) >= {'total', 'size', 'alignment', 'spacing', 'shape', 'grid'}


# --- End-to-end on a synthetic card ----------------------------------------
# A small generated card exercises the real CLI in a second or two, so the
# folder-lifecycle contract is covered without the gigapixel scan.

import subprocess
import sys

import pyvips

REPO = Path(__file__).resolve().parent


def make_card(path, cols=4, rows=3):
    """White page rectangles on a dark card, laid out on a grid."""
    a = np.zeros((1500, 2000), 'uint8')
    for r in range(rows):
        for c in range(cols):
            y = 200 + r * 420
            x = 60 + c * 480
            a[y:y + 340, x:x + 400] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(path))
    return cols * rows


def run_segmenter(*args):
    return subprocess.run([sys.executable, str(REPO / "segment_microfiche.py"), *args],
                          capture_output=True, text=True, cwd=str(REPO))


def real_pages(out):
    """Page files excluding the header prepage (page zero)."""
    from segment_microfiche import HEADER_PAGE_STEM as _h
    return [p for p in (out / "pages").glob("page_*.tif") if p.stem != _h]


def test_end_to_end_writes_the_card_contract(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out))
    assert proc.returncode == 0, proc.stderr

    assert sorted(p.name for p in out.iterdir()) == [
        "_debug", DONE_SENTINEL, "page_coordinates.csv", "pages"]
    assert len(real_pages(out)) == n
    assert "Card Quality:" in (out / "page_coordinates.csv").read_text().splitlines()[0]


def test_rerun_clears_stale_pages_end_to_end(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    # --no-archive: re-running needs the panorama to stay put.
    assert run_segmenter("-i", str(src), "-O", str(out), "--no-archive").returncode == 0
    stale = out / "pages" / "page_099.tif"
    stale.write_bytes(b"leftover")

    assert run_segmenter("-i", str(src), "-O", str(out), "--no-archive").returncode == 0
    assert not stale.exists()
    assert len(real_pages(out)) == n


def test_skip_extraction_does_not_destroy_existing_pages(tmp_path):
    """Inspection mode must not eat a finished card's output."""
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out), "--no-archive").returncode == 0
    assert len(real_pages(out)) == n

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")
    assert proc.returncode == 0, proc.stderr
    assert len(real_pages(out)) == n, "pages were deleted"
    assert (out / DONE_SENTINEL).exists(), "sentinel removed from a still-valid card"


def test_visualization_is_written_without_debug_flag(tmp_path):
    """The operator's ground truth: one image showing what was found, in which
    order - available on every run, offline, without re-running anything.
    Heavy artifacts (the full-res binary) stay behind --debug."""
    src = tmp_path / "612130000012_00012.jpg"
    make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out), "--no-archive").returncode == 0
    assert (out / "_debug" / "visualization.jpg").exists()
    assert not (out / "_debug" / "temp_binary.tif").exists()


def test_failed_detection_leaves_the_visualization(tmp_path):
    """A no-pages failure is exactly when the picture matters most."""
    src = tmp_path / "612130000012_00012.jpg"
    # Et kort med BARE header og ingen sider: den lyse massen overst gir Otsu
    # en ekte terskel (et nesten-uniformt bilde gir terskel 0, som gjor ALT til
    # forgrunn), men headermasken fjerner den - og da er det ingenting igjen.
    a = np.zeros((1500, 2000), 'uint8')
    a[0:90, :] = 255          # 6% < headerskippens 8%
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")
    assert proc.returncode == 2, proc.stderr
    assert (out / "_debug" / "visualization.jpg").exists()
    assert not (out / DONE_SENTINEL).exists()


def test_missing_input_exits_1_not_2(tmp_path):
    """Exit codes are the app's contract: 2 means "no pages detected". A run
    without --input must fail with the generic 1, so the app never mistakes an
    operator mistake for an empty card."""
    proc = run_segmenter()
    assert proc.returncode == 1
    assert "--input is required" in proc.stderr


def test_near_uniform_card_fails_loudly(tmp_path):
    """Otsu on a (nearly) uniform surface returns threshold 0, so EVERYTHING
    becomes foreground and the whole card comes out as one giant "page" -
    something wrong that looks normal. That is not a card: fail exactly like
    the no-pages case - exit 2, source to error/, viz written, never a _done."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.full((1500, 2000), 40, 'uint8')
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (out / DONE_SENTINEL).exists()
    assert (out / "_debug" / "visualization.jpg").exists()
    assert (tmp_path / "error" / src.name).exists()
    assert not (out / "pages").exists() or not list((out / "pages").iterdir())


def test_all_foreground_card_fails_loudly(tmp_path):
    """A blank bright scan with a few dark specks gives Otsu a real threshold,
    but ~everything lands above it: same one-giant-page failure as the uniform
    case, just with a nonzero threshold. Foreground share near 100% is not a
    card - pages always sit on visible card background."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.full((1500, 2000), 200, 'uint8')
    a[700:720, 500:520] = 10
    a[1200:1215, 1600:1620] = 10
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (out / DONE_SENTINEL).exists()
    assert (tmp_path / "error" / src.name).exists()


def test_debug_artifacts_stay_out_of_the_card_folder(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out), "--debug").returncode == 0
    assert (out / "_debug" / "visualization.jpg").exists()
    loose = [p.name for p in out.iterdir() if p.suffix.lower() in ('.jpg', '.tif', '.tiff')]
    assert loose == [], f"image files loose in the card folder: {loose}"


# --- Reading order -----------------------------------------------------------
# Journals are always read left-to-right, then top-to-bottom (like text).
# Page numbering is the downstream contract: a wrong default scrambles every
# multi-column card silently.

from segment_microfiche import READING_ORDER


def test_reading_order_defaults_to_rows():
    assert READING_ORDER == 'rows'


def test_default_numbering_walks_the_top_row_first(tmp_path):
    """With 4 columns x 3 rows, pages 1-4 must share the top row."""
    src = tmp_path / "612130000012_00012.jpg"
    make_card(src, cols=4, rows=3)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")
    assert proc.returncode == 0, proc.stderr

    rows = [line.split(",") for line
            in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    ys = [int(r[2]) for r in rows[:4]]
    assert max(ys) - min(ys) < 200, f"pages 1-4 are not one row: y={ys}"
    xs = [int(r[1]) for r in rows[:4]]
    assert xs == sorted(xs), "top row is not numbered left-to-right"


# --- Size-outlier rejection -------------------------------------------------
# Stitching can leave a bright band along a card edge. It survives the
# minimum-size filter (it is huge, not small) and lands in the page list as a
# blank strip, shifting every later page number by one.
#
# The filter targets SHAPE, not size (2026-09-03): real journals hold pages of
# genuinely different sizes, and losing a page costs more than gaining a blank
# crop. Only band-shaped detections are dropped - grossly oversized in one
# dimension while at-or-under the median in the other.

from segment_microfiche import (BAND_RATIO, DEFAULT_PADDING_RATIO,
                                drop_band_detections)


def test_default_crop_margin_is_generous():
    """3% margin (2026-09-03): unclear edges on real journals should err
    toward including a little card background, never toward cutting text."""
    assert DEFAULT_PADDING_RATIO == 0.03


def _grid(n=20, w=2040, h=1630):
    return [(100 + (i % 5) * 2100, 200 + (i // 5) * 1700, w, h) for i in range(n)]


def test_drops_a_wide_flat_edge_strip():
    """The real case: 33208 x 732 against a 2040 x 1630 median."""
    boxes = _grid() + [(1110, 24770, 33208, 732)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == [(1110, 24770, 33208, 732)]
    assert len(kept) == 20


def test_keeps_normally_varying_pages():
    boxes = [(100 + i * 2100, 200, 2040 + (i % 7) * 12, 1630 - (i % 5) * 9)
             for i in range(20)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert kept == boxes


def test_keeps_a_short_page_with_normal_width():
    """A half-height page is a plausible journal page (receipts, notes).

    Until 2026-09-03 this was dropped as an outlier. Real journals hold pages
    of varying sizes; a wrongly kept blank costs one extra crop, a wrongly
    dropped page silently loses journal content.
    """
    boxes = _grid() + [(500, 900, 2040, 700)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert (500, 900, 2040, 700) in kept


def test_keeps_widely_varying_page_sizes():
    """Half-size to median-size pages on one card, all kept."""
    boxes = _grid() + [(500, 900, 1100, 900), (2700, 900, 1500, 1200)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert len(kept) == 22


def test_drops_a_tall_narrow_strip():
    """The vertical twin of the edge band."""
    boxes = _grid() + [(50, 100, 600, 24000)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == [(50, 100, 600, 24000)]


def test_keeps_a_merged_row_of_pages():
    """Weak page edges can fuse a whole row into one detection: 3-4 pages wide
    but FULL page height. That is merged content, not a stitch band - dropping
    it silently loses every page in the row (reported by Trond 2026-09-03).
    A real band is a thin sliver (documented: 720 high vs 1630 median, 0.44x);
    width cannot discriminate, since a full-width band and a fully merged
    16-page row are equally wide. Height is the tell."""
    merged_row = (100, 200, 3 * 2040 + 2 * 60, 1630)
    boxes = _grid() + [merged_row]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert merged_row in kept


def test_still_drops_a_sliver_band_at_page_multiple_width():
    """The discriminator must be thinness, not width: a sliver exactly as wide
    as 3 pages is still a band, because no page is 0.4x the median height."""
    sliver = (100, 24770, 3 * 2040 + 2 * 60, 700)
    boxes = _grid() + [sliver]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == [sliver]


def test_a_merged_row_is_kept_and_warned_about_end_to_end(tmp_path):
    """Weak edges fuse a row at detect scale. The row must survive as ONE
    detection (content present, inspectable in the viz) and the log must warn
    loudly, so inspection mode shows the problem without squinting at boxes."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.zeros((1500, 2000), 'uint8')
    for r in range(3):
        for c in range(4):
            y, x = 200 + r * 420, 60 + c * 480
            a[y:y + 340, x:x + 400] = 255
    # Top row: bridge the gaps so thresholding fuses it into one blob.
    a[200:540, 60:60 + 3 * 480 + 400] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")
    assert proc.returncode == 0, proc.stderr
    rows = [line for line in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    assert len(rows) == 9, rows  # 8 single pages + the fused row, nothing dropped
    # NB: tmp_path contains "merged" (pytest names it after the test) and
    # stdout prints paths, so match the warning phrase, not the bare word.
    assert "suspected merged pages" in proc.stdout.lower()


def test_keeps_contours_aligned_with_kept_boxes():
    boxes = _grid(6) + [(0, 0, 33208, 732)]
    contours = [f"c{i}" for i in range(7)]
    kept, kept_contours, dropped = drop_band_detections(boxes, contours, BAND_RATIO)
    assert len(kept) == len(kept_contours) == 6
    assert kept_contours == [f"c{i}" for i in range(6)]
    assert len(dropped) == 1


def test_does_not_filter_when_the_median_is_untrustworthy():
    """If most boxes would be dropped, the median is junk — keep everything.

    Nothing here agrees with anything else, so the median describes no real
    page. Discarding "outliers" would throw away most of the card.
    """
    boxes = [(0, 0, 100, 100), (0, 0, 1000, 1000),
             (0, 0, 5000, 5000), (0, 0, 9000, 9000)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert kept == boxes


def test_too_few_boxes_to_judge_are_left_alone():
    boxes = [(0, 0, 2040, 1630), (0, 0, 33208, 732)]
    kept, _, dropped = drop_band_detections(boxes, None, BAND_RATIO)
    assert dropped == []
    assert kept == boxes


# --- Header prepage ---------------------------------------------------------
# The masked header band carries the card's only identifying metadata (title,
# part number, date, and "N of M" card index). It is kept as page zero: sorts
# ahead of page_001, scaled right down, and easy to drop downstream.

from segment_microfiche import HEADER_PAGE_STEM, HEADER_PROXY_SCALE


def test_writes_the_header_as_page_zero(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out), "--header-page").returncode == 0

    header = out / "pages" / f"{HEADER_PAGE_STEM}.tif"
    assert header.exists(), "header prepage not written"
    assert len(real_pages(out)) == n


def test_header_prepage_is_scaled_right_down(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"
    run_segmenter("-i", str(src), "-O", str(out), "--header-page")

    header = pyvips.Image.new_from_file(str(out / "pages" / f"{HEADER_PAGE_STEM}.tif"))
    page = pyvips.Image.new_from_file(str(out / "pages" / "page_001.tif"))
    # Source card is 2000px wide; the band spans full width before scaling.
    assert header.width == int(2000 * HEADER_PROXY_SCALE)
    assert header.width < page.width


def test_header_prepage_sorts_before_the_first_page():
    """The OCR app orders pages by the last integer in the stem."""
    import re
    stems = [f"{HEADER_PAGE_STEM}", "page_001", "page_002", "page_010"]
    keys = [int(re.findall(r"\d+", s)[-1]) for s in stems]
    assert keys == sorted(keys)
    assert keys[0] == 0


def test_no_header_prepage_when_header_skip_is_zero(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out), "--header-page",
                         "--header-skip", "0").returncode == 0
    assert not (out / "pages" / f"{HEADER_PAGE_STEM}.tif").exists()
    assert len(real_pages(out)) == n


def test_header_prepage_is_on_by_default(tmp_path):
    """The import side handles page zero as of 2026-08-23, so it ships."""
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out)).returncode == 0
    assert (out / "pages" / f"{HEADER_PAGE_STEM}.tif").exists()
    assert len(list((out / "pages").glob("page_*.tif"))) == n + 1


def test_header_prepage_can_be_suppressed(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    n = make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out),
                         "--no-header-page").returncode == 0
    assert not (out / "pages" / f"{HEADER_PAGE_STEM}.tif").exists()
    assert len(list((out / "pages").glob("page_*.tif"))) == n


def test_header_page_flag_still_accepted(tmp_path):
    """The app may already pass --header-page. It must not become an error:
    argparse exits 2 on an unknown flag, which collides with EXIT_NO_PAGES."""
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--header-page")
    assert proc.returncode == 0, proc.stderr
    assert (out / "pages" / f"{HEADER_PAGE_STEM}.tif").exists()


# --- Archiving the source panorama -----------------------------------------
# Panoramas/ is a work queue: once a card is segmented its panorama moves to
# PanoramaArchive/ so what remains in Panoramas/ is what still needs doing.

from segment_microfiche import ARCHIVE_DIR_NAME, move_without_clobber


def test_move_without_clobber_relocates(tmp_path):
    src = tmp_path / "612130000012_00016.tif"
    src.write_bytes(b"panorama")
    dest = move_without_clobber(src, tmp_path / ARCHIVE_DIR_NAME)
    assert not src.exists()
    assert dest.read_bytes() == b"panorama"
    assert dest.parent.name == ARCHIVE_DIR_NAME


def test_move_without_clobber_keeps_an_existing_file(tmp_path):
    dest_dir = tmp_path / ARCHIVE_DIR_NAME
    dest_dir.mkdir()
    (dest_dir / "card.tif").write_bytes(b"earlier")

    src = tmp_path / "card.tif"
    src.write_bytes(b"later")
    dest = move_without_clobber(src, dest_dir)

    assert (dest_dir / "card.tif").read_bytes() == b"earlier"
    assert dest.read_bytes() == b"later"
    assert dest.name != "card.tif"


def test_panorama_is_archived_after_a_successful_run(tmp_path):
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out)).returncode == 0

    assert not src.exists(), "panorama left in the work queue"
    archived = tmp_path / ARCHIVE_DIR_NAME / "612130000012_00016.jpg"
    assert archived.exists(), "panorama not in the archive"
    assert (out / DONE_SENTINEL).exists()


def test_archiving_can_be_turned_off(tmp_path):
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000012_00016.jpg"
    make_card(src)

    assert run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--no-archive").returncode == 0
    assert src.exists(), "panorama archived despite --no-archive"


def test_skip_extraction_never_archives(tmp_path):
    """Inspection mode must not move the operator's source."""
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000012_00016.jpg"
    make_card(src)

    assert run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction").returncode == 0
    assert src.exists(), "inspection mode moved the source"


def test_a_failed_card_goes_to_error_not_the_archive(tmp_path):
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000999_00016.jpg"
    # Specks only: blobs exist but none survive the minimum-page-size filter.
    a = np.zeros((1500, 2000), 'uint8')
    for yy in range(100, 1400, 300):
        for xx in range(100, 1900, 400):
            a[yy:yy + 12, xx:xx + 12] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))

    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert (panoramas / "error" / src.name).exists(), "not moved to error/"
    assert not (tmp_path / ARCHIVE_DIR_NAME).exists(), "failed card was archived"
