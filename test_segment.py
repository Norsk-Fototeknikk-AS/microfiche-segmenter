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
    assert q['grid'] == "1 row: 1"


def test_quality_reports_a_grid_for_no_boxes():
    q = compute_card_quality([], None)
    assert q['grid'] == "0 rows"


def test_grid_string_reports_rows_and_their_counts():
    """The cards have rows, not a grid: 'no columns' (Trond, 2026-09-04).
    Report what is real - row count and pages per row."""
    boxes = [(100 + c * 500, 100 + r * 400, 400, 300)
             for r in range(3) for c in range(4)]
    q = compute_card_quality(boxes, None)
    assert q['grid'] == "3 rows: 4+4+4"


def test_unaligned_rows_are_not_penalized():
    """Rows start where they start - pages in different rows are NOT
    vertically aligned on real cards. Column alignment must not drag the
    score down for a correctly detected card."""
    aligned = [(100 + c * 500, 100 + r * 400, 400, 300)
               for r in range(3) for c in range(4)]
    shifted = [(100 + c * 500 + r * 230, 100 + r * 400, 400, 300)
               for r in range(3) for c in range(4)]
    q_aligned = compute_card_quality(aligned, None)
    q_shifted = compute_card_quality(shifted, None)
    assert q_shifted['alignment'] == q_aligned['alignment']
    assert q_shifted['total'] > 80, q_shifted


def test_irregular_spacing_within_a_row_still_scores_lower():
    """In-row spacing is the real spacing axis and must still be measured."""
    regular = [(100 + c * 500, 100, 400, 300) for c in range(5)]
    irregular = [(100, 100, 400, 300), (620, 100, 400, 300),
                 (1400, 100, 400, 300), (1850, 100, 400, 300),
                 (2700, 100, 400, 300)]
    assert (compute_card_quality(irregular, None)['spacing']
            < compute_card_quality(regular, None)['spacing'])


def test_quality_always_carries_the_keys_main_prints():
    for boxes in ([], [(0, 0, 10, 10)], [(0, 0, 10, 10), (20, 0, 10, 10)]):
        q = compute_card_quality(boxes, None)
        assert set(q) >= {'total', 'size', 'alignment', 'spacing', 'shape', 'grid'}


# --- Card geometry limits ----------------------------------------------------
# Domain truth (Trond, 2026-09-04): at most 5 rows, 11 pages per row, and no
# column structure at all. Tighter caps are misdetection tripwires.

from segment_microfiche import MAX_PAGES_PER_ROW, MAX_ROWS


def test_card_limits_match_the_physical_cards():
    """Trond 2026-09-08: rows of 12 pages measured in production, wants
    margin - hence 13 (was 11)."""
    assert MAX_ROWS == 5
    assert MAX_PAGES_PER_ROW == 13


def test_too_many_pages_in_a_row_warns(tmp_path):
    """One page beyond the limit must warn; derived from the constant so a
    raised limit keeps this test honest."""
    n = MAX_PAGES_PER_ROW + 1
    src = tmp_path / "612130000012_00012.jpg"
    width = 120 + n * 520
    a = np.zeros((800, width), 'uint8')
    for c in range(n):
        a[200:540, 60 + c * 520:460 + c * 520] = 255
    pyvips.Image.new_from_memory(a.tobytes(), width, 800, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"
    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--no-invert")
    assert proc.returncode == 0, proc.stderr
    assert f"{n} pages in one row" in proc.stdout, proc.stdout


def test_too_many_rows_warns(tmp_path):
    src = tmp_path / "612130000012_00012.jpg"
    a = np.zeros((3000, 1200), 'uint8')
    for r in range(6):
        for c in range(2):
            a[100 + r * 480:440 + r * 480, 100 + c * 500:500 + c * 500] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 1200, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"
    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--no-invert", "--header-skip", "0")
    assert proc.returncode == 0, proc.stderr
    assert "6 rows" in proc.stdout, proc.stdout


# --- End-to-end on a synthetic card ----------------------------------------
# A small generated card exercises the real CLI in a second or two, so the
# folder-lifecycle contract is covered without the gigapixel scan.

import shutil
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


def make_inverted_card(path, cols=4, rows=3):
    """Dark page rectangles on a BRIGHT card - the real journal card type
    (fiche negatives in a light jacket), first seen 2026-09-04. Detection
    needs --invert here."""
    a = np.full((1500, 2000), 230, 'uint8')
    for r in range(rows):
        for c in range(cols):
            y = 200 + r * 420
            x = 60 + c * 480
            a[y:y + 340, x:x + 400] = 25
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(path))
    return cols * rows


def test_invert_finds_dark_pages_on_bright_card(tmp_path):
    """--invert was never exercised until the first real journal card arrived
    (2026-09-04) and turned out dark-on-bright. It was broken: the pyvips
    threshold already yields 0/255, and the pipeline's *255 "conversion" wraps
    255 to 1 in uint8, so bitwise_not turns BOTH levels nonzero - everything
    becomes foreground and the run dies on the degenerate-threshold guard."""
    src = tmp_path / "612130000012_00012.jpg"
    n = make_inverted_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--invert")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [line for line in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    assert len(rows) == n, proc.stdout


def test_card_structure_touching_the_border_is_not_pages(tmp_path):
    """The real journal card (2026-09-04) is a light jacket with a dark frame
    and dark edge-to-edge stripes between rows. Inverted, that structure is one
    connected foreground component ENCLOSING every page - RETR_EXTERNAL sees
    only the frame, and the pages inside it vanish. Structure always touches
    the image border; pages never do. Border-connected foreground must be
    removed, leaving exactly the pages."""
    src = tmp_path / "612130000012_00012.jpg"
    # Proportions matter, at DETECT scale: real stripes are ~1-2% of image
    # height - thin enough to be structure, thick enough to survive the detect
    # erosion. A small fixture cannot represent both, so this one is big.
    a = np.full((12000, 16000), 230, 'uint8')
    a[:800, :] = 20; a[-800:, :] = 20; a[:, :800] = 20; a[:, -800:] = 20  # frame
    n = 0
    for r in range(3):
        y = 1000 + r * 2400
        a[y:y + 200, :] = 30                        # stripe, edge to edge
        top = y + 150 if r == 0 else y + 300        # row 0 OVERLAPS its stripe,
        for c in range(4):                          # like the real card's sleeve
            x = 1000 + c * 3600
            a[top:y + 2100, x:x + 3000] = 25        # dark pages
            n += 1
    pyvips.Image.new_from_memory(a.tobytes(), 16000, 12000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--invert", "--header-skip", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [line for line in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    assert len(rows) == n, proc.stdout


# --- Splitting touching pages ------------------------------------------------
# Weak edges merge touching pages into one detection. The gap between two real
# pages is a projection VALLEY: columns (or rows) where the foreground share
# drops far below the box's typical level. Measured on the real card
# (2026-09-04): the valley between the two tape-pages is 24 full-res px wide
# at 45% foreground against 99% inside the pages.

from segment_microfiche import find_projection_valleys


def test_finds_the_measured_real_valley():
    """A profile shaped like the real card's: ~0.99 everywhere, one narrow dip
    to ~0.45."""
    share = np.full(430, 0.99)
    share[209:214] = 0.45          # the 24px valley at 20% scan scale
    assert find_projection_valleys(share, min_gap=3) == [211]


def test_a_shallow_dip_is_not_a_valley():
    """Content variation inside a page (sparse text, dust) must not split it."""
    share = np.full(430, 0.95)
    share[200:220] = 0.80
    assert find_projection_valleys(share, min_gap=3) == []


def test_valleys_touching_the_ends_are_edges_not_gaps():
    """A low run at the box edge is the box boundary itself."""
    share = np.full(430, 0.99)
    share[0:15] = 0.1
    share[420:430] = 0.1
    assert find_projection_valleys(share, min_gap=3) == []


def test_two_valleys_split_a_triple_merge():
    share = np.full(600, 0.98)
    share[195:205] = 0.3
    share[395:405] = 0.3
    assert find_projection_valleys(share, min_gap=3) == [199, 399]


def test_a_valley_narrower_than_min_gap_is_noise():
    share = np.full(430, 0.99)
    share[210] = 0.2
    assert find_projection_valleys(share, min_gap=3) == []


def _dirty_seam(a, x, y, w, h):
    """A real-world page gap: not clean background, but a mix (the tapes on
    the real card overlap their gap - 45% foreground). Thresholded at full
    resolution the mix reads as page (so detection merges the neighbors);
    averaged first (resize, then threshold) it reads as background (so the
    split scan sees the valley). Striped rows give exactly that duality."""
    for row in range(y, y + h):
        if row % 5 < 3:
            a[row, x:x + w] = 25       # dark rows: 60% duty, fine-grained
        else:
            a[row, x:x + w] = 230


def test_splits_two_pages_sharing_a_dirty_seam(tmp_path):
    src = tmp_path / "pair.jpg"
    a = np.full((4000, 8000), 230, 'uint8')
    a[500:3500, 1000:4000] = 25                    # page A
    a[500:3500, 4024:7024] = 25                    # page B
    _dirty_seam(a, 4000, 500, 24, 3000)            # the 24px mixed gap
    pyvips.Image.new_from_memory(a.tobytes(), 8000, 4000, 1, 'uchar').write_to_file(str(src))

    from segment_microfiche import split_box_by_projection
    # otsu 90: the seam's resize-average (0.6*25 + 0.4*230 = 107) must land
    # on the LIGHT side for the scan, while per-pixel thresholding at full
    # res keeps its dark rows as page - the duality that merges detection.
    out = split_box_by_projection(str(src), (1000, 500, 6024, 3000),
                                  otsu_thresh=90, invert=True,
                                  min_w=100, min_h=100)
    assert len(out) == 2, out
    (ax, ay, aw, ah), (bx, by, bw, bh) = sorted(out)
    assert abs(ax - 1000) < 60 and abs(ax + aw - 4012) < 60, out
    assert abs(bx - 4012) < 60 and abs(bx + bw - 7024) < 60, out
    assert ay == by == 500 and ah == bh == 3000, out


def test_edge_artifact_cuts_do_not_veto_the_real_split(tmp_path):
    """Erosion-compensated boxes carry a rim of background at their edges,
    which reads as a shallow valley just inside the border. Such a cut would
    create an impossibly small piece - discard THAT cut alone, never the real
    mid-box cut alongside it (the real card's pair went unsplit exactly this
    way: a col-7 edge artifact vetoed the col-430 gap)."""
    src = tmp_path / "pair.jpg"
    a = np.full((4000, 8000), 230, 'uint8')
    a[500:3500, 1000:4000] = 25
    a[500:3500, 4024:7024] = 25
    _dirty_seam(a, 4000, 500, 24, 3000)
    # A dark sliver inside the rim (jacket edge/shadow) separates the rim's
    # low columns from the box edge, so they read as an internal valley.
    a[500:3500, 952:958] = 25
    pyvips.Image.new_from_memory(a.tobytes(), 8000, 4000, 1, 'uchar').write_to_file(str(src))

    from segment_microfiche import split_box_by_projection
    # Box deliberately 60px wider on the left: the strip of background inside
    # the box edge yields an artifact valley there.
    out = split_box_by_projection(str(src), (940, 500, 6084, 3000),
                                  otsu_thresh=90, invert=True,
                                  min_w=500, min_h=500)
    assert len(out) == 2, out


def test_a_single_page_is_not_split(tmp_path):
    src = tmp_path / "single.jpg"
    a = np.full((4000, 8000), 230, 'uint8')
    a[500:3500, 1000:4000] = 25
    pyvips.Image.new_from_memory(a.tobytes(), 8000, 4000, 1, 'uchar').write_to_file(str(src))

    from segment_microfiche import split_box_by_projection
    box = (1000, 500, 3000, 3000)
    out = split_box_by_projection(str(src), box, otsu_thresh=125, invert=True,
                                  min_w=100, min_h=100)
    assert out == [box]


def test_merged_pair_is_split_end_to_end(tmp_path):
    """Detection merges the seam-sharing pair; the split pass must separate
    them again, so the CSV carries every page and numbering stays honest."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.full((12000, 16000), 230, 'uint8')
    a[:800, :] = 20; a[-800:, :] = 20; a[:, :800] = 20; a[:, -800:] = 20
    n = 0
    for r in range(3):
        y = 1000 + r * 2400
        a[y:y + 200, :] = 30
        for c in range(4):
            x = 1000 + c * 3600
            a[y + 300:y + 2100, x:x + 3000] = 25
            n += 1
    # Row 1: close the gap between pages 1 and 2 with a dirty seam so the
    # detect pass merges them (columns 4000..7024 shift: page2 moved left).
    y = 1000 + 1 * 2400
    a[y + 300:y + 2100, 4600:7600] = 230           # erase original page 2
    a[y + 300:y + 2100, 4024:7024] = 25            # rebuild it against the seam
    _dirty_seam(a, 4000, y + 300, 24, 1800)
    pyvips.Image.new_from_memory(a.tobytes(), 16000, 12000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [line.split(",") for line in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    assert len(rows) == n, proc.stdout
    assert "Split" in proc.stdout, proc.stdout
    # The split pair sits in row 2 (pages 5 and 6 in reading order): two
    # boxes, not one double-wide.
    widths = sorted(int(r[3]) for r in rows)
    assert widths[-1] < 3600, widths                # nothing double-wide left
# --- Polarity autodetection --------------------------------------------------
# The app sends no flags, and the two known card types have opposite polarity
# (Yamaha: bright pages on dark card; journals: dark pages on light jacket).
# Border sampling cannot tell them apart - the dark mounting surround frames
# BOTH types, so the border ring reads background either way (measured on the
# real card 2026-09-04). What does discriminate is physics: no single page can
# span ~the whole card, so the wrong polarity yields full-width row boxes and
# the right one yields floating page-sized boxes.

from segment_microfiche import autodetect_inversion, page_likeness_score


def _binary_of(name):
    img = pyvips.Image.new_from_file(str(REPO / "testdata" / name)).colourspace('b-w')
    a = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                   shape=[img.height, img.width])
    t, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((a >= t) * 255).astype(np.uint8)


def test_score_counts_page_sized_boxes_and_penalizes_row_sized():
    W, H = 2000, 1500
    pages = [(100, 100, 300, 250), (500, 100, 300, 250)]
    rows = [(50, 500, 1900, 250)]                 # 95% of width: not a page
    assert page_likeness_score(pages, W, H) == 2
    assert page_likeness_score(pages + rows, W, H) == 1
    assert page_likeness_score(rows, W, H) == -1


def test_autodetect_inverts_the_real_journal_card():
    b = _binary_of("real_card_10pct.jpg")
    h, w = b.shape
    invert = autodetect_inversion(b, int(h * 0.08), int(w * 0.02), int(h * 0.02))
    assert invert is True


def test_autodetect_keeps_normal_polarity_on_a_yamaha_type_card():
    a = np.zeros((1500, 2000), 'uint8')
    for r in range(3):
        for c in range(4):
            a[200 + r * 420:540 + r * 420, 60 + c * 480:460 + c * 480] = 255
    invert = autodetect_inversion(a.copy(), 0, 40, 30)
    assert invert is False


def test_autodetect_prefers_nothing_over_impossible_boxes_on_the_blank():
    """The blank jacket is not a tie: normal polarity reads the empty bright
    rows as six full-width "pages" (score -6) - silent junk that would have
    become a _done'd card. Inverted yields nothing (score 0), which downstream
    turns into the loud no-pages exit. Choosing emptiness over impossible
    boxes is the point of the penalty."""
    b = _binary_of("real_card_blank_10pct.jpg")
    h, w = b.shape
    invert = autodetect_inversion(b, int(h * 0.08), int(w * 0.02), int(h * 0.02))
    assert invert is True


def test_autodetect_tie_prefers_inverted():
    """When neither polarity yields anything (a scoreless tie), prefer
    inverted: the production default is the journal card type - dark pages on
    a light jacket (Trond, 2026-09-04)."""
    b = np.zeros((500, 800), np.uint8)
    assert autodetect_inversion(b, 0, 20, 20) is True


def test_autodetect_runs_end_to_end_without_flags(tmp_path):
    """The app sends no polarity flag; an inverted journal-type card must come
    out right anyway, and the log must SAY the polarity was auto-chosen - a
    silent guess is the trap."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.full((12000, 16000), 230, 'uint8')
    a[:800, :] = 20; a[-800:, :] = 20; a[:, :800] = 20; a[:, -800:] = 20
    n = 0
    for r in range(3):
        y = 1000 + r * 2400
        a[y:y + 200, :] = 30
        for c in range(4):
            x = 1000 + c * 3600
            a[y + 300:y + 2100, x:x + 3000] = 25
            n += 1
    pyvips.Image.new_from_memory(a.tobytes(), 16000, 12000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [line for line in (out / "page_coordinates.csv").read_text().splitlines()
            if line and not line.startswith(("#", "page"))]
    assert len(rows) == n, proc.stdout
    assert "Auto-detected polarity: INVERTING" in proc.stdout


def test_no_invert_disables_autodetection(tmp_path):
    """--no-invert is the manual override the other way: polarity is forced
    normal and autodetection must not even run."""
    src = tmp_path / "612130000012_00012.jpg"
    make_inverted_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--no-invert")
    assert "Auto-detected" not in proc.stdout


def test_invert_and_no_invert_together_exit_1():
    proc = run_segmenter("-i", "whatever.jpg", "--invert", "--no-invert")
    assert proc.returncode == 1
    assert "mutually exclusive" in proc.stderr


# --- Real journal card (committed at detect scale) --------------------------
# testdata/ holds the first real journal card (2026-09-04, no patient info) at
# 10% scale - exactly what the detect pass sees. Two variants: the jacket with
# two anonymized pages (black tape at accurate page size/position - geometry is
# truth, texture is not), and the SAME jacket empty. The empty card is the
# negative control: everything on it is structure, so detection must find
# nothing at all.

from segment_microfiche import (DETECT_ERODE_ITERATIONS, DETECT_ERODE_KERNEL,
                                MIN_PAGE_HEIGHT_RATIO, MIN_PAGE_WIDTH_RATIO,
                                clear_border_connected)


def _detect_on_committed_thumb(name):
    """The detect pass, replicated on an image already at detect scale."""
    img = pyvips.Image.new_from_file(str(REPO / "testdata" / name)).colourspace('b-w')
    a = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                   shape=[img.height, img.width])
    thresh, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = ((a >= thresh) * 255).astype(np.uint8)
    b = cv2.bitwise_not(b)                       # dark pages on light jacket
    h, w = b.shape
    hdr = int(h * 0.08)
    b[:hdr, :] = 0
    from segment_microfiche import remove_structure_rows
    remove_structure_rows(b, int(h * MIN_PAGE_HEIGHT_RATIO), hdr)
    kernel = np.ones((DETECT_ERODE_KERNEL,) * 2, np.uint8)
    b = cv2.erode(b, kernel, iterations=DETECT_ERODE_ITERATIONS)
    clear_border_connected(b)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return [bx for bx in boxes
            if bx[2] >= int(w * MIN_PAGE_WIDTH_RATIO)
            and bx[3] >= int(h * MIN_PAGE_HEIGHT_RATIO)]


def test_real_blank_card_detects_nothing():
    """Everything on the empty jacket is structure - stripes, frame, header."""
    assert _detect_on_committed_thumb("real_card_blank_10pct.jpg") == []


def test_real_card_finds_the_taped_pages():
    """The two tape-pages touch each other, so until splitting exists they are
    ONE detection - at the tape position, top-right of row one."""
    boxes = _detect_on_committed_thumb("real_card_10pct.jpg")
    assert len(boxes) == 1, boxes
    x, y, w, h = boxes[0]
    assert 2100 < x < 2250 and 200 < y < 300, boxes    # detect-scale pixels
    assert 380 < w < 480 and 250 < h < 350, boxes      # two pages wide, one high


def test_header_page_reaches_down_to_the_first_page_row(tmp_path):
    """On the real journal card the typed header text sits BELOW the 8% mask
    line - a fixed-ratio page_000 crop cuts the date and card index in half.
    The header is by definition everything above the first page row, so the
    crop must extend to the topmost detected page (capped at twice the
    configured band, so a sparse card cannot swallow empty rows into it)."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.full((12000, 16000), 230, 'uint8')
    a[:800, :] = 20; a[-800:, :] = 20; a[:, :800] = 20; a[:, -800:] = 20
    for r in range(3):
        y = 1000 + r * 2400
        a[y:y + 200, :] = 30
        for c in range(4):
            x = 1000 + c * 3600
            a[y + 300:y + 2100, x:x + 3000] = 25
    pyvips.Image.new_from_memory(a.tobytes(), 16000, 12000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--invert", "--no-archive")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    header = pyvips.Image.new_from_file(str(out / "pages" / "page_000.tif"))
    # First page row starts at y=1300 (10.8% of 12000); the 8% mask alone
    # would cut at 960. At 1/16 scale: >= 1240/16 = 77, old behavior 60.
    assert header.height >= 77, header.height
    assert header.height <= int(12000 * 0.08 * 2) // 16 + 1, header.height


# --- Structure-row removal ---------------------------------------------------
# The real journal cards (2026-09-04) are light jackets with dark edge-to-edge
# stripes between rows, plus dark bands along the header and the bottom.
# Inverted, that structure connects to everything it touches - including pages
# whose sleeves overlap a stripe - so connectivity alone cannot separate them.
# Full-width row-runs ARE separable: a run is structure if it is thinner than
# any possible page, or touches the image boundary / header mask; a (merged)
# row of pages is page-height and floats inside the card.

from segment_microfiche import remove_structure_rows


def _canvas(h=1000, w=2000):
    return np.zeros((h, w), np.uint8)


def test_removes_a_thin_full_width_stripe():
    b = _canvas()
    b[500:520, :] = 255           # 2% of height, edge to edge
    b[100:250, 300:500] = 255     # a page, for contrast
    removed = remove_structure_rows(b, min_page_h=40, top_boundary=0)
    assert b[510, 1000] == 0
    assert b[150, 400] == 255
    assert removed > 0


def test_keeps_a_tall_full_width_run_in_the_middle():
    """A fully merged row of pages is full-width too - height tells them apart."""
    b = _canvas()
    b[400:600, :] = 255           # 20% of height: page-height, floating
    remove_structure_rows(b, min_page_h=40, top_boundary=0)
    assert b[500, 1000] == 255


def test_removes_a_tall_run_touching_the_bottom():
    b = _canvas()
    b[900:1000, :] = 255          # tall, but runs into the image edge
    remove_structure_rows(b, min_page_h=40, top_boundary=0)
    assert b[950, 1000] == 0


def test_removes_a_tall_run_at_the_header_boundary():
    """The header mask cuts structure mid-band; what abuts the cut is the
    band's continuation, not a page row."""
    b = _canvas()
    b[80:200, :] = 255            # starts right at the header mask line
    remove_structure_rows(b, min_page_h=40, top_boundary=80)
    assert b[150, 1000] == 0


def test_coverage_oscillating_at_the_threshold_is_not_shredded():
    """A page band whose coverage straddles the threshold row by row (noise)
    must be judged as ONE band, not as dozens of 1-row 'stripes' that each
    fall under the height floor and get deleted - shredding real pages."""
    b = _canvas()
    b[300:500, 0:1695] = 255                       # a page band at 84.75%
    b[300:500:2, 1695:1706] = 255                  # alternate rows: 85.3%
    removed = remove_structure_rows(b, min_page_h=40, top_boundary=0)
    assert removed == 0
    assert (b[300:500, 0:1695] == 255).all()


def test_page_rows_are_never_touched():
    """Rows holding separated pages have big gaps - far below full coverage."""
    b = _canvas()
    for c in range(4):
        b[100:300, 100 + c * 500:400 + c * 500] = 255
    removed = remove_structure_rows(b, min_page_h=40, top_boundary=0)
    assert removed == 0
    assert b[200, 200] == 255


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


# --- Anonymized visualization (--anon-viz) --------------------------------
# m4-studio holds journal content that must never leave the machine. The anon
# viz replaces every detected blob with a SOLID filled silhouette so seams and
# geometry stay visible while no glyph survives.

from segment_microfiche import make_anon_mask


def test_anon_mask_fills_content_holes():
    """Text inside a page is a hole in the blob - it must be filled solid."""
    img = np.zeros((200, 300), np.uint8)
    img[40:160, 50:250] = 255
    img[90:100, 100:150] = 0  # a glyph-sized hole of readable content
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = make_anon_mask(img.shape, contours, dilate_radius=0)

    assert mask[95, 120] == 255, "content hole leaked through the silhouette"
    assert set(np.unique(mask)) <= {0, 255}, "mask must be strictly two-level"


def test_anon_mask_preserves_through_gaps():
    """A light seam splitting a page reaches the blob edge - it is NOT a hole
    and must stay visible: the gap is the diagnostic signal we ship out."""
    img = np.zeros((300, 200), np.uint8)
    img[20:100, 30:170] = 255   # top third
    img[120:280, 30:170] = 255  # bottom two thirds
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = make_anon_mask(img.shape, contours, dilate_radius=0)

    assert not mask[100:120, 30:170].any(), "seam gap was filled shut"
    assert mask[60, 100] == 255 and mask[200, 100] == 255


def test_anon_mask_dilation_undoes_detection_erosion():
    """Contours come from the eroded image; the silhouette grows back by the
    erosion radius so it lines up with the (compensated) page boxes."""
    img = np.zeros((200, 200), np.uint8)
    img[50:150, 50:150] = 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = make_anon_mask(img.shape, contours, dilate_radius=6)

    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (44, 155, 44, 155)


from segment_microfiche import render_anon_viz


def test_anon_viz_render_is_two_level_plus_overlay_colors():
    """The safety property: no pixel value derived from card CONTENT may
    survive - only pure black/white silhouette plus the overlay palette.
    Any midtone means source texture leaked into the shipped image."""
    mask = np.zeros((1650, 2200), np.uint8)  # wider than the 2000px viz cap
    mask[200:800, 300:1900] = 255
    boxes_fullres = [(3000, 2000, 7000, 6000), (11000, 2000, 7000, 6000)]

    viz = render_anon_viz(mask, boxes_fullres, fullres_to_mask=0.1,
                          label="Card Quality: 99/100 (GOOD)  |  1 row: 2",
                          banner_color=(0, 180, 0))

    colors = {tuple(c) for c in np.unique(viz.reshape(-1, 3), axis=0)}
    allowed = {(0, 0, 0), (255, 255, 255),   # silhouette
               (0, 255, 0), (0, 0, 255),     # box outline, page number
               (30, 30, 30), (0, 180, 0)}    # banner background, banner text
    assert colors <= allowed, f"unexpected midtones leaked: {colors - allowed}"
    assert (0, 255, 0) in colors, "page boxes missing from overlay"
    assert (0, 180, 0) in colors, "banner missing from overlay"


def test_anon_viz_flag_writes_anonymized_view(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--anon-viz",
                         "--no-archive")
    assert proc.returncode == 0, proc.stderr

    assert (out / "_debug" / "anon_viz.jpg").exists(), "anon viz not written"
    assert (out / "_debug" / "visualization.jpg").exists(), \
        "normal viz must still be written (it stays on the machine)"


def test_anon_viz_not_written_without_flag(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    out = tmp_path / "card"

    assert run_segmenter("-i", str(src), "-O", str(out),
                         "--no-archive").returncode == 0
    assert not (out / "_debug" / "anon_viz.jpg").exists()


def test_real_card_anon_mask_holds_no_glyph_sized_detail():
    """Safety pin on the real journal card: every silhouette in the anon mask
    is at least page-sized. Only size-filtered detections may be drawn - if a
    future change feeds unfiltered contours in, stray glyph silhouettes would
    be readable text and this must fail."""
    from segment_microfiche import detect_page_boxes, make_anon_mask
    b = cv2.bitwise_not(_binary_of("real_card_10pct.jpg"))
    h, w = b.shape
    min_w, min_h = int(w * MIN_PAGE_WIDTH_RATIO), int(h * MIN_PAGE_HEIGHT_RATIO)

    boxes, contours, b, _ = detect_page_boxes(b, int(h * 0.08), min_w, min_h)
    assert boxes, "fixture regression: the tape pages were not detected"

    mask = make_anon_mask(
        b.shape, contours,
        erosion_radius(DETECT_ERODE_KERNEL, DETECT_ERODE_ITERATIONS))

    assert set(np.unique(mask)) <= {0, 255}
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    assert n > 1, "mask is empty"
    for i in range(1, n):
        assert stats[i, cv2.CC_STAT_WIDTH] >= min_w, "sub-page-sized silhouette"
        assert stats[i, cv2.CC_STAT_HEIGHT] >= min_h, "sub-page-sized silhouette"


# --- Fragment guard (exit 3) ----------------------------------------------
# Production 2026-09-07: ~half the successfully segmented cards had pages cut
# horizontally in two detections (top ~1/3 + bottom ~2/3) - suspected light
# stitching seams in the panorama. Half-pages archived as success with shifted
# numbering is the worst kind of quiet corruption, and content may be MISSING
# in the gap, so merging is wrong: the card must fail loudly instead.

from segment_microfiche import (EXIT_SUSPECT_FRAGMENTS, expected_page_height,
                                find_fragment_groups)


def _whole_page_row(n=4, w=400, h=600, pitch=500, y=100):
    return [(i * pitch, y, w, h) for i in range(n)]


def test_fragment_groups_flags_a_third_two_thirds_split():
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),   # top third
        (2000, 310, 400, 390),   # bottom two thirds, 30px gap
    ]
    assert find_fragment_groups(boxes) == [(4, 5)]


def test_fragment_groups_ignores_whole_pages_in_adjacent_rows():
    """Vertically stacked WHOLE pages align in x and sit close - but their
    union is ~2 pages tall, nowhere near the expected page height."""
    boxes = _whole_page_row(y=100) + _whole_page_row(y=780)  # 80px row gap
    assert find_fragment_groups(boxes) == []


def test_fragment_groups_ignores_side_by_side_pages():
    assert find_fragment_groups(_whole_page_row()) == []


def test_expected_height_survives_half_the_boxes_being_fragments():
    """Median would sink toward the fragments; the upper quartile stays on the
    whole pages as long as fragments are a minority of... up to ~75%."""
    whole = _whole_page_row(n=4)
    frags = [(2000, 100, 400, 180), (2000, 310, 400, 390),
             (2500, 100, 400, 200), (2500, 330, 400, 370)]
    exp = expected_page_height(whole + frags)
    assert 550 <= exp <= 620, exp


def test_expected_height_caps_a_vertically_merged_outlier():
    """One unsplittable vertical merge (~2 pages tall) must not drag the
    estimate up to double height - that would make whole-page row stacks
    match the union band and fail good cards."""
    boxes = _whole_page_row() + [(2000, 100, 400, 1280)]
    assert expected_page_height(boxes) <= 900


def test_real_card_with_synthetic_seam_flags_a_fragment_pair():
    """The production signature, reproduced on the fasit: a light stitching
    seam at 1/3 page height cuts the detection in two stacked boxes. Worst
    case on purpose - EVERY detection on this card is a fragment, so there is
    no whole page left to anchor the expected height."""
    from segment_microfiche import detect_page_boxes
    b = cv2.bitwise_not(_binary_of("real_card_10pct.jpg"))
    h, w = b.shape
    b[244 + 301 // 3: 244 + 301 // 3 + 8, :] = 0  # seam through the tape pair

    boxes, _, _, _ = detect_page_boxes(
        b, int(h * 0.08), int(w * MIN_PAGE_WIDTH_RATIO),
        int(h * MIN_PAGE_HEIGHT_RATIO))
    boxes = sorted(boxes, key=lambda bb: bb[1])

    assert len(boxes) == 2, boxes
    assert find_fragment_groups(boxes) == [(0, 1)]


def make_seamed_card(path, seams=1, seam_rows=(1,)):
    """A 4x3 card where light seams cut every page of the given rows - the
    defective input seen in production (1 seam = 1/3+2/3 pairs,
    2 seams = stacks of three).

    Proportions matter: pages must be tall enough that a 1/3 fragment
    survives the detect-scale erosion (radius 6 there = 120px here, per
    side), and the seam wide enough (30px = 3px at detect scale) not to
    average back into foreground in the downsample."""
    a = np.zeros((3000, 2000), 'uint8')
    for r in range(3):
        for c in range(4):
            y = 350 + r * 900
            x = 60 + c * 480
            a[y:y + 700, x:x + 400] = 255
    for r in seam_rows:
        for k in range(1, seams + 1):
            seam_y = 350 + r * 900 + (700 * k) // (seams + 1)
            a[seam_y:seam_y + 30, :] = 0
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 3000, 1, 'uchar').write_to_file(str(path))


def test_seamed_card_is_geometry_completed(tmp_path):
    """Phase 2 contract flip (Trond, 2026-09-08): content is intact, so
    grid-matching fragment pairs MERGE into whole pages instead of exit 3.
    The card completes with the merged pages marked and counted."""
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000012_00016.jpg"
    make_seamed_card(src)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / DONE_SENTINEL).exists()
    assert "4 pages geometry-completed" in proc.stdout, proc.stdout
    assert len(real_pages(out)) == 12, "4 merged + 8 whole pages"


def test_whole_card_still_passes_the_fragment_guard(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--no-archive")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_anon_viz_marks_fragment_pairs_in_orange_and_stays_clean():
    mask = np.zeros((500, 800), np.uint8)
    mask[100:400, 100:700] = 255
    boxes = [(1000, 1000, 3000, 1000), (1000, 2200, 3000, 2200),
             (5000, 1000, 3000, 3400)]

    viz = render_anon_viz(mask, boxes, fullres_to_mask=0.1,
                          label="SUSPECT FRAGMENTS", banner_color=(0, 0, 200),
                          fragment_indices={0, 1})

    colors = {tuple(c) for c in np.unique(viz.reshape(-1, 3), axis=0)}
    allowed = {(0, 0, 0), (255, 255, 255), (0, 255, 0), (0, 0, 255),
               (30, 30, 30), (0, 0, 200), (0, 165, 255)}
    assert colors <= allowed, f"unexpected midtones leaked: {colors - allowed}"
    assert (0, 165, 255) in colors, "fragment boxes not marked in orange"
    assert (0, 255, 0) in colors, "ordinary page box lost its green"


# --- RAPPORT.command / rapport.py -----------------------------------------
# Safe report extraction on the air-gapped m4-studio: inspect every panorama
# in a folder, collect ONLY anonymized artifacts for the USB stick. Journal
# content (visualization.jpg, binaries, page crops) must never end up in the
# report folder, not even by accident - hence a hard whitelist.

import pytest

import rapport


def test_whitelist_allows_only_anonymous_artifacts():
    assert rapport.is_safe_artifact("anon_viz.jpg")
    assert rapport.is_safe_artifact("612130000012_00001 Panorama_rapport.txt")
    assert rapport.is_safe_artifact("SAMMENDRAG.txt")

    assert not rapport.is_safe_artifact("visualization.jpg")
    assert not rapport.is_safe_artifact("page_001.tif")
    assert not rapport.is_safe_artifact("temp_binary.tif")
    assert not rapport.is_safe_artifact("612130000012_00001 Panorama.jpg")


def test_guarded_copy_refuses_files_outside_the_whitelist(tmp_path):
    src = tmp_path / "visualization.jpg"
    src.write_bytes(b"journal content")
    with pytest.raises(rapport.UnsafeArtifact):
        rapport.copy_safe(src, tmp_path / "out" / "visualization.jpg")
    assert not (tmp_path / "out").exists()


def test_summary_lines_show_status_exit_pages_and_fragments():
    ok = rapport.summary_line("kort_a", 0, 16, 0)
    frag = rapport.summary_line("kort_b", 3, 16, 2)
    fail = rapport.summary_line("kort_c", 2, 0, 0)
    assert ok.startswith("OK") and " 16 " in ok and "kort_a" in ok
    assert frag.startswith("FRAGMENT") and "2 grupper" in frag
    assert fail.startswith("FEIL") and "exit 2" in fail


def test_summary_warns_on_low_quality_even_at_exit_zero():
    """Field data 2026-09-07: every seam-sick card scored below 50, every
    healthy one above 74. A low score on an OK card is the early warning."""
    low = rapport.summary_line("kort_d", 0, 16, 0, quality=23.6)
    fine = rapport.summary_line("kort_a", 0, 16, 0, quality=98.9)
    assert "LAV KVALITET" in low and "23.6" in low
    assert "LAV KVALITET" not in fine


def test_quality_is_parsed_from_run_output():
    out = "...\n  Card Quality: 23.6/100  (POOR)\n..."
    assert rapport.parse_quality(out) == 23.6
    assert rapport.parse_quality("no quality here") is None


def test_rapport_end_to_end_collects_only_safe_artifacts(tmp_path):
    """Two panoramas - one clean, one seamed - inspected into a report folder
    that must hold nothing but whitelisted files and name the seamed card."""
    src = tmp_path / "arkiv"
    src.mkdir()
    make_card(src / "612130000012_00001.jpg")
    make_seamed_card(src / "612130000012_00002.jpg")

    report_dir = rapport.run_report(src, tmp_path / "RAPPORT-test",
                                    open_finder=False)

    files = sorted(p.name for p in report_dir.iterdir())
    assert "SAMMENDRAG.txt" in files
    assert "612130000012_00001_rapport.txt" in files
    for name in files:
        assert rapport.is_safe_artifact(name), f"unsafe file leaked: {name}"

    summary = (report_dir / "SAMMENDRAG.txt").read_text()
    assert "612130000012_00002" in summary and "FRAGMENT" in summary
    assert "612130000012_00001" in summary
    # anon viz per card came along, under whitelisted names
    assert any(n.endswith("anon_viz.jpg") for n in files), files


def test_failed_card_still_writes_anon_viz(tmp_path):
    """Production 2026-09-07: error-path cards got a text report but no
    anonymized image - and the failing cards are exactly the ones that most
    need to be SEEN across the air gap. Same principle as visualization.jpg:
    written on EVERY run, failure included."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.zeros((1500, 2000), 'uint8')
    for yy in range(100, 1400, 300):
        for xx in range(100, 1900, 400):
            a[yy:yy + 12, xx:xx + 12] = 255  # specks only: exit 2, no pages
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 1500, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--anon-viz")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert (out / "_debug" / "anon_viz.jpg").exists(), \
        "error path skipped the anonymized view"


def test_rapport_flags_a_card_without_anon_viz_as_failure(tmp_path):
    """A missing anonymized image must never look like success in the
    summary - whatever the reason it is missing."""
    src = tmp_path / "arkiv"
    src.mkdir()
    (src / "612130000012_00099.jpg").write_bytes(b"not an image at all")

    report_dir = rapport.run_report(src, tmp_path / "RAPPORT-test",
                                    open_finder=False)

    summary = (report_dir / "SAMMENDRAG.txt").read_text()
    line = next(l for l in summary.splitlines() if "612130000012_00099" in l)
    assert line.startswith("FEIL"), line
    assert "anon_viz mangler" in line, line


# --- Chain extension of the fragment guard --------------------------------
# Production card 612130000111_00012 passed as OK: pages split in STACKS of
# 3-4 fragments, and the pairwise union of any two neighbours lands BELOW the
# 0.8x band. The guard must group stacked boxes transitively and judge the
# chain's union against the expected page height.

def test_fragment_groups_catches_a_three_way_split():
    """Two seams through one page: three stacked fragments. Any PAIR of them
    unions below the band - only the full chain reaches page height."""
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),
        (2000, 295, 400, 190),
        (2000, 500, 400, 180),
    ]
    assert find_fragment_groups(boxes) == [(4, 5, 6)]


def test_fragment_groups_still_reports_plain_pairs():
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),
        (2000, 310, 400, 390),
    ]
    assert find_fragment_groups(boxes) == [(4, 5)]


def test_fragment_chain_stops_before_swallowing_the_next_row():
    """Transitive chaining may link a fragment stack to a whole page in the
    row below it when the gap is tight; the group must still be found as the
    contiguous window that hits page height - not lost because the full
    chain's union overshoots the band."""
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),
        (2000, 295, 400, 190),
        (2000, 500, 400, 180),
        (2000, 710, 400, 600),   # whole page, next row, 30px gap
    ]
    assert find_fragment_groups(boxes) == [(4, 5, 6)]


def test_fragment_groups_ignores_whole_pages_in_adjacent_rows_too():
    boxes = _whole_page_row(y=100) + _whole_page_row(y=780)
    assert find_fragment_groups(boxes) == []


def test_triple_seamed_card_is_geometry_completed(tmp_path):
    """Stacks of three merge just like pairs."""
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000111_00012.jpg"
    make_seamed_card(src, seams=2)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--no-archive")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "4 pages geometry-completed" in proc.stdout, proc.stdout
    assert len(real_pages(out)) == 12


def test_card_needing_too_much_repair_still_fails(tmp_path):
    """Two of three rows seamed: geometry would have to save 8 of 12 pages.
    A card THAT sick is not repaired into silent success - exit 3 stands.
    (Threshold is preliminary; field worst case so far is 28%.)"""
    panoramas = tmp_path / "Panoramas"
    panoramas.mkdir()
    src = panoramas / "612130000012_00016.jpg"
    make_seamed_card(src, seam_rows=(1, 2))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out))

    assert proc.returncode == EXIT_SUSPECT_FRAGMENTS, proc.stdout + proc.stderr
    assert not (out / DONE_SENTINEL).exists()
    assert "geometry" in (proc.stdout + proc.stderr).lower()
    assert (panoramas / "error" / src.name).exists(), "not moved to error/"


def test_short_document_is_extended_end_to_end(tmp_path):
    """A short document in a full row gets the full sheet and the card
    completes - no guard trip, visible in the log."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.zeros((3000, 2000), 'uint8')
    for r in range(3):
        for c in range(4):
            y = 350 + r * 900
            x = 60 + c * 480
            h = 300 if (r, c) == (1, 1) else 700   # one short document
            a[y:y + h, x:x + 400] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 2000, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 pages geometry-completed" in proc.stdout, proc.stdout
    rows_csv = (out / "page_coordinates.csv").read_text().splitlines()[2:]
    heights = sorted({int(r.split(",")[4]) for r in rows_csv})
    assert max(heights) - min(heights) <= 130, \
        f"short page not extended to row height: {heights}"


# --- Illumination-robust thresholding --------------------------------------
# Production 2026-09-08: panoramas came out MOTTLED (patchy brightness after a
# machine upgrade) with content intact and readable. The global Otsu threshold
# put patches on the wrong side -> swiss-cheese binaries, fragment guard fired
# on every card (correctly - the binary WAS bad). Fix: estimate the
# low-frequency illumination field (per-cell high percentile tracks the
# bright class), flatten before thresholding. Clean images are ~unaffected.

from segment_microfiche import (ILLUM_WARN_SHARE, estimate_illumination_field,
                                illumination_plan)


def make_journal_card(path, mottled=False):
    """Low-contrast journal-type card (dark pages on a light jacket), with
    REAL proportions: 12 pages per row, so a page is narrower than an
    illumination-field cell - on physical cards a page is ~1/14 of the card
    width, which is what lets the field's per-cell p90 track the jacket. The
    mottle is a multiplicative low-frequency field strong enough that the
    darkest jacket dips below the brightest page - measured to silently eat
    pages under the old global threshold."""
    h, w = 1500, 6400
    a = np.full((h, w), 180, 'float32')
    for r in range(3):
        for c in range(12):
            y = 200 + r * 420
            x = 60 + c * 520
            a[y:y + 340, x:x + 400] = 110
    if mottled:
        # Production mottling lives at stitch-tile scale (a panorama is 4x4
        # tiles), so the blob size is relative to WIDTH on both axes - the
        # field tracks patches of that scale, not arbitrarily sharp ones.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = np.ones((h, w), np.float32)
        for cx, cy, s, sign in [(0.25, 0.3, 0.22, -0.45),
                                (0.7, 0.75, 0.28, +0.25)]:
            f += sign * np.exp(-(((xx - cx * w) ** 2 + (yy - cy * h) ** 2)
                                 / (2 * (s * w) ** 2)))
        f += 0.1 * (xx / w - 0.5)
        a = a * f
    a = np.clip(a, 0, 255).astype('uint8')
    pyvips.Image.new_from_memory(a.tobytes(), w, h, 1, 'uchar').write_to_file(str(path))


def test_mottled_journal_card_segments_like_the_clean_one(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_journal_card(src, mottled=True)
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Found 36 potential pages" in proc.stdout, proc.stdout
    assert "uneven illumination" in proc.stdout.lower(), \
        "operator must be told the card was mottled"


def test_clean_journal_card_gets_no_illumination_warning(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_journal_card(src, mottled=False)

    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Found 36 potential pages" in proc.stdout, proc.stdout
    assert "uneven illumination" not in proc.stdout.lower()


def _mottle_thumb(a):
    """The synthetic mottle from the coordinator's spec, on the real fasit."""
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = np.ones((h, w), np.float32)
    for cx, cy, s, sign in [(0.78, 0.15, 0.22, -0.5), (0.3, 0.7, 0.3, +0.3)]:
        f += sign * np.exp(-(((xx / w - cx) ** 2 + (yy / h - cy) ** 2)
                             / (2 * s ** 2)))
    f += 0.15 * (xx / w - 0.5)
    return np.clip(a.astype(np.float32) * f, 0, 255).astype(np.uint8)


def test_mottled_real_card_detects_the_tape_pages():
    """Replica of the main pass on the committed fasit with a dark blotch
    right over the tape pages: plan from a small thumb, threshold surface at
    full (thumb) resolution, then the ordinary detect pipeline."""
    img = pyvips.Image.new_from_file(
        str(REPO / "testdata" / "real_card_10pct.jpg")).colourspace('b-w')
    gray = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                      shape=[img.height, img.width])
    gray = _mottle_thumb(gray)
    h, w = gray.shape

    from segment_microfiche import detect_page_boxes
    small = cv2.resize(gray, (w // 10, h // 10), interpolation=cv2.INTER_AREA)
    field, norm, thresh, share = illumination_plan(small)
    assert share > ILLUM_WARN_SHARE, "this mottle must trip the warning"

    surface = cv2.resize(field * (thresh / norm), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    b = cv2.bitwise_not(((gray >= surface) * 255).astype(np.uint8))
    boxes, _, _, _ = detect_page_boxes(
        b, int(h * 0.08), int(w * MIN_PAGE_WIDTH_RATIO),
        int(h * MIN_PAGE_HEIGHT_RATIO))
    boxes = sorted(boxes, key=lambda bb: (bb[1], bb[0]))

    # The clean fasit detects the tape pair at (2167, 244, 426, 301); the
    # blotch sits right on top of it. One merged box or the two pages split
    # at the tape gap are both faithful detections.
    assert 1 <= len(boxes) <= 2, boxes
    x0 = min(bb[0] for bb in boxes)
    x1 = max(bb[0] + bb[2] for bb in boxes)
    assert abs(x0 - 2167) <= 15 and abs(x1 - 2593) <= 15, boxes
    assert all(abs(bb[1] - 247) <= 15 and abs(bb[3] - 297) <= 15
               for bb in boxes), boxes


def test_env_versions_are_logged_at_startup(tmp_path):
    """Every rapport.txt from the air-gapped machine must document the
    environment it ran in (environment drift was once a suspected culprit)."""
    src = tmp_path / "612130000012_00016.jpg"
    make_card(src)
    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction")
    assert proc.returncode == 0
    assert "Env: python " in proc.stdout, proc.stdout
    for lib in ("numpy", "opencv", "pyvips"):
        assert lib in proc.stdout, proc.stdout


# --- VIS-PANORAMA.command / vis_panorama.py --------------------------------
# Preview cannot open the zstd-TIFF panoramas on m4-studio; Trond needs
# LOSSLESS viewing copies (Digital Color Meter on real pixel values) written
# next to the sources. On-machine only - journal data stays put.

import vis_panorama


def test_view_path_never_overwrites(tmp_path):
    src = tmp_path / "612130000012_00001 Panorama.tif"
    src.write_bytes(b"x")
    first = vis_panorama.view_path(src)
    assert first.name == "612130000012_00001 Panorama_visning.tif"
    first.write_bytes(b"existing")
    second = vis_panorama.view_path(src)
    assert second.name == "612130000012_00001 Panorama_visning-2.tif"
    assert second.parent == src.parent


def test_sources_takes_files_and_direct_folder_children(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "b.tiff").write_bytes(b"x")
    (tmp_path / "c.jpg").write_bytes(b"x")          # not a panorama TIFF
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.tif").write_bytes(b"x")           # never subfolders

    from_folder = vis_panorama.sources([str(tmp_path)])
    assert [p.name for p in from_folder] == ["a.tif", "b.tiff"]

    single = vis_panorama.sources([str(tmp_path / "a.tif")])
    assert [p.name for p in single] == ["a.tif"]


def test_convert_writes_a_lossless_lzw_copy(tmp_path):
    """Digital Color Meter on the copy must read the SOURCE's pixel values.
    (Source uses deflate here - the local libvips lacks zstd write support -
    but the conversion path is identical, and m4-studio's pyvips provably
    reads the production zstd panoramas: the segmenter does.)"""
    rng = np.random.RandomState(7)
    a = rng.randint(0, 255, (64, 80), dtype=np.uint8)
    src = tmp_path / "612130000012_00001 Panorama.tif"
    pyvips.Image.new_from_memory(a.tobytes(), 80, 64, 1, 'uchar').write_to_file(
        str(src), compression='deflate')

    dst = vis_panorama.convert(src)

    assert dst.name.endswith("_visning.tif") and dst.parent == tmp_path
    out = pyvips.Image.new_from_file(str(dst))
    b = np.ndarray(buffer=out.write_to_memory(), dtype=np.uint8, shape=[64, 80])
    assert np.array_equal(a, b), "viewing copy must be lossless"


# --- Phase 2: geometric completion ------------------------------------------
# Decided by Trond 2026-09-08: stitching is fixed, content is INTACT - the
# remaining defects are material (washed-out patches at jacket brightness,
# short documents, half-dark pages). The sheet size is known, so geometry
# overrides the binary: grid-matching fragment chains are MERGED into one
# page box (crops are cut from the original graytone anyway), lone short
# detections are extended to their row's height. What does not reconcile
# with the grid still exits 3 - the guard gets a repair step, not a
# weakening. Field verdict from 28 production groups: every one resolves
# by sheet size (x-IoU 1.0, gap 0, union ~page height).

from segment_microfiche import (GEOMETRY_MAX_INVENTED_SHARE,
                                GEOMETRY_MAX_REPAIR_SHARE, complete_geometry)


def test_geometry_merges_a_fragment_chain_into_one_page():
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),
        (2000, 295, 400, 190),
        (2000, 500, 400, 180),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    assert len(new) == 5
    repaired = [b for b, f in zip(new, flags) if f]
    assert repaired == [(2000, 100, 400, 580)]
    assert any("merged 3 fragments" in n for n in notes)


def test_geometry_refuses_a_merge_that_invents_too_much():
    """A chain at the very edge of the group criteria - maximal gaps AND
    maximal x offsets - would invent over a third of the 'page'. That is
    fabrication, not repair."""
    boxes = _whole_page_row() + [
        (2000, 100, 360, 120),
        (2040, 300, 360, 120),
        (2000, 500, 360, 140),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == [(4, 5, 6)]
    assert not any(flags), "refused fragments must not be marked repaired"
    assert len(new) == 7
    assert any("REFUSED" in n for n in notes)


def test_geometry_extends_a_short_document_to_row_height():
    """A short document in a full row gets the full sheet: worst case is a
    little empty film in the crop."""
    boxes = [(0, 100, 400, 600), (500, 100, 400, 600),
             (1000, 105, 400, 250),   # short document
             (1500, 100, 400, 600)]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    extended = [b for b, f in zip(new, flags) if f]
    assert extended == [(1000, 100, 400, 600)]
    assert any("extended short detection" in n for n in notes)


def test_geometry_leaves_a_short_box_without_full_neighbours_alone():
    """No row anchor - nothing to extend toward. The guard downstream still
    sees the true geometry."""
    boxes = [(0, 100, 400, 250), (500, 105, 400, 260)]
    new, flags, _, _ = complete_geometry(boxes)
    assert new == boxes
    assert not any(flags)


def test_geometry_handles_the_worst_field_card():
    """Card 612130000135 from production 2026-09-08 (34 detections, 7 groups,
    half-dark pages, strip fragments): all seven in-band chains merge, none
    refused, and the repair share stays under the card threshold."""
    boxes = [
        (1920, 3790, 880, 2210), (4220, 4070, 800, 1900),
        (10640, 3870, 810, 2020), (17150, 3950, 1240, 1860),
        (19330, 4210, 1000, 1580), (19510, 3480, 730, 850),
        (21490, 4260, 1350, 1500), (25860, 4610, 1950, 1100),
        (2010, 6710, 1870, 435), (2010, 7145, 1870, 1485),
        (4190, 6680, 1350, 2760), (5540, 6680, 720, 2760),
        (6370, 6650, 1950, 1875), (8560, 6630, 2040, 1875),
        (10720, 6590, 1830, 1880), (12900, 6540, 2040, 2790),
        (15080, 7090, 1280, 1745), (17260, 6950, 2020, 2330),
        (19420, 6470, 1700, 510), (19420, 6980, 1700, 1345),
        (23790, 6940, 1300, 1470), (25970, 7100, 1730, 2050),
        (2010, 8630, 1870, 830), (6370, 8525, 1950, 885),
        (8560, 8505, 2040, 885), (10720, 8470, 1830, 890),
        (15080, 8835, 1280, 465), (19420, 8325, 1700, 925),
        (21990, 8330, 890, 870), (23790, 8410, 1300, 770),
        (19260, 10170, 1680, 455), (19260, 10625, 1680, 2075),
        (21440, 11950, 1380, 730), (24110, 11950, 1140, 700),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    repaired = sum(flags)
    assert repaired >= 7, notes
    assert repaired <= GEOMETRY_MAX_REPAIR_SHARE * len(new), \
        "the worst real card must still pass the card threshold"
    merged_heights = [b[3] for b, f in zip(new, flags) if f]
    for h in merged_heights:
        assert h <= 3000, f"a repaired page taller than any real page: {h}"


def test_anon_viz_marks_repaired_pages_in_blue_and_stays_clean():
    from segment_microfiche import GEOMETRY_MARK_COLOR
    mask = np.zeros((500, 800), np.uint8)
    mask[100:400, 100:700] = 255
    boxes = [(1000, 1000, 3000, 3400), (5000, 1000, 3000, 3400)]

    viz = render_anon_viz(mask, boxes, fullres_to_mask=0.1,
                          label="1 geometry-completed",
                          banner_color=(0, 180, 0),
                          repaired_indices={1})

    colors = {tuple(c) for c in np.unique(viz.reshape(-1, 3), axis=0)}
    allowed = {(0, 0, 0), (255, 255, 255), (0, 255, 0), (0, 0, 255),
               (30, 30, 30), (0, 180, 0), GEOMETRY_MARK_COLOR}
    assert colors <= allowed, f"unexpected midtones leaked: {colors - allowed}"
    assert GEOMETRY_MARK_COLOR in colors, "repaired page not marked in blue"
    assert (0, 255, 0) in colors


# --- Vertical stripe merging (Trond's override, 2026-09-08) -----------------
# A page split into two full-height STRIPS has the outline of one page, and
# the format guarantees uniform page sizes - same safety as horizontally.
# The whole risk is merging two real neighbour pages; their union is ~2x the
# page width PLUS a real gap, so the 0.8-1.2x width band excludes them.

from segment_microfiche import find_stripe_groups


def test_stripe_groups_finds_the_field_cards_split_page():
    """Card 612130000135 detections 11+12 (real coordinates): one page as two
    full-height strips, flanked by whole pages from the same row."""
    boxes = [
        (4190, 6680, 1350, 2760), (5540, 6680, 720, 2760),   # the strips
        (12900, 6540, 2040, 2790), (17260, 6950, 2020, 2330),
        (25970, 7100, 1730, 2050),
    ]
    assert find_stripe_groups(boxes) == [(0, 1)]


def test_stripe_groups_never_merges_real_neighbour_pages():
    """THE critical safety property: a full row of ordinary pages at field
    pitch (2040 wide, 140px gaps) must produce no stripe groups - their
    union is ~2x a page wide."""
    boxes = [(2000 + i * 2180, 3000, 2040, 2790) for i in range(6)]
    assert find_stripe_groups(boxes) == []


def test_geometry_merges_vertical_stripes():
    boxes = [
        (4190, 6680, 1350, 2760), (5540, 6680, 720, 2760),
        (12900, 6540, 2040, 2790), (17260, 6950, 2020, 2330),
        (25970, 7100, 1730, 2050),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    repaired = [b for b, f in zip(new, flags) if f]
    assert repaired == [(4190, 6680, 2070, 2760)]
    assert any("vertical stripes" in n for n in notes)


def test_geometry_leaves_a_real_page_row_untouched():
    boxes = [(2000 + i * 2180, 3000, 2040, 2790) for i in range(6)]
    new, flags, _, _ = complete_geometry(boxes)
    assert sorted(new) == sorted(boxes)
    assert not any(flags)


def test_vertical_seam_card_is_geometry_completed_end_to_end(tmp_path):
    """A jacket-level vertical seam through one page: the two strips merge
    back into one page and the card completes."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.full((3000, 6400), 180, 'uint8')
    for r in range(3):
        for c in range(6):
            y = 350 + r * 900
            x = 100 + c * 1050
            a[y:y + 700, x:x + 1000] = 110
    seam_x = 100 + 1050 + 400   # through page (row 1, col 1)
    a[350 + 900:350 + 900 + 700, seam_x:seam_x + 30] = 180
    pyvips.Image.new_from_memory(a.tobytes(), 6400, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "vertical stripes" in proc.stdout, proc.stdout
    assert "1 pages geometry-completed" in proc.stdout, proc.stdout
    rows_csv = (out / "page_coordinates.csv").read_text().splitlines()[2:]
    assert len(rows_csv) == 18, rows_csv


# --- Phase 3: sub-band chains (Trond, 2026-09-08) ---------------------------
# Field-confirmed (pair 17+27 in RAPPORT-2026-09-08-3, union 0.79x): chains
# meeting the x-IoU and gap criteria but with union BELOW 0.8x expected are
# pages that LOST height to a defect - they must repair too, not slip
# through unflagged. The lower union bound is removed for MERGING (the upper
# ~1.8x stays as the cross-row guard), and the merged short result is
# completed to the row's anchors. Row-boundary safety: field row gaps are
# ~840-920px vs the 418px gap criterion - a short page can never link
# across the row boundary.


def test_sub_band_chain_completes_to_row_height_field_case():
    """Pair 17+27 from card 612130000135 (real coordinates): a half-dark
    page detected as 1745+465 with union 2210 = 0.79x expected. One page,
    repaired, completed to the row's median height."""
    boxes = [
        (15080, 7090, 1280, 1745), (15080, 8835, 1280, 465),   # the pair
        (12900, 6540, 2040, 2790), (17260, 6950, 2020, 2330),  # row anchors
        (25970, 7100, 1730, 2050),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    assert len(new) == 4, ("the pair must merge into ONE page - a sibling "
                           "fragment left behind becomes a ghost page", new)
    repaired = [b for b, f in zip(new, flags) if f]
    assert len(repaired) == 1, (new, flags)
    x, y, w, h = repaired[0]
    assert (x, w) == (15080, 1280)
    assert h >= 2500, f"sub-band chain not completed to row height: {h}"
    assert any("row height" in n for n in notes), notes


def test_page_that_lost_a_third_of_its_height_repairs():
    """Synthetic: two pieces totalling 70% of the page with a small gap -
    union 0.7x is below the old band and used to slip through silently."""
    boxes = _whole_page_row() + [
        (2000, 100, 400, 180),
        (2000, 335, 400, 185),
    ]
    new, flags, notes, refused = complete_geometry(boxes)

    assert refused == []
    assert len(new) == 5, ("both pieces must be consumed by the repair", new)
    repaired = [b for b, f in zip(new, flags) if f]
    assert repaired == [(2000, 100, 400, 600)], (repaired, notes)


def test_short_page_never_links_across_the_row_boundary():
    """Field geometry: the row gap (~920px) is far beyond the gap criterion
    (15% of 2790 = 418px). A bottom-of-row fragment must not chain onto the
    next row's page even with the lower union bound removed."""
    boxes = [
        (19420, 8325, 1700, 925),     # bottom fragment, row 2
        (19260, 10170, 1680, 2500),   # whole page, row 3 (gap 920)
        (12900, 6540, 2040, 2790), (17260, 6950, 2020, 2330),
    ]
    assert find_fragment_groups(boxes, union_min=0.0) == []
    new, flags, _, _ = complete_geometry(boxes)
    assert (19260, 10170, 1680, 2500) in new, "row-3 page was consumed"
    merged_tall = [b for b, f in zip(new, flags) if f and b[3] > 3000]
    assert merged_tall == [], f"cross-row merge happened: {merged_tall}"


# --- Page-size prior and grid snapping (architecture shift, 2026-09-08) -----
# Trond: "igjen feiler den med sider som er feil størrelse" - the page size
# is a KNOWN CONSTANT of the format, not something to derive per blob. Every
# accepted detection snaps to a full page box: the blob gives position, the
# prior gives size. The merge/extension passes become special cases of the
# snap. Pitch (regular within a row) resolves collisions; what cannot be
# reconciled with the grid still exits 3.

from segment_microfiche import (PAGE_SIZE_PRIOR, PAGE_SIZE_TOLERANCE,
                                resolve_page_size, snap_pages)


def test_page_size_comes_from_the_prior_tuned_by_best_detections():
    """Detections near the prior tune it (bounded); strips do not vote."""
    boxes = [(2010, 6710, 2040, 2790), (4190, 6680, 2040, 2760),
             (6370, 6650, 2050, 2800),
             (8560, 6630, 880, 2210), (10720, 6590, 800, 1900)]  # strips
    pw, ph, note = resolve_page_size(boxes)
    assert abs(pw - 2043) <= 10, pw
    assert abs(ph - 2790) <= 20, ph
    # and always inside the bound around the prior
    assert abs(pw - PAGE_SIZE_PRIOR[0]) <= PAGE_SIZE_TOLERANCE * PAGE_SIZE_PRIOR[0]


def test_page_size_falls_back_to_estimate_off_format():
    """A card whose pages are nowhere near the prior (test fixtures, other
    formats): per-card estimate, loudly noted - never silent garbage."""
    boxes = [(i * 500, 100, 400, 600) for i in range(5)]
    pw, ph, note = resolve_page_size(boxes)
    assert 380 <= pw <= 420 and 570 <= ph <= 630, (pw, ph)
    assert note is not None and "prior" in note.lower()


def test_snap_gives_field_strips_full_page_boxes():
    """Card 612130000135 row 1 (real coordinates): narrow washed-out strips,
    NO full-height anchor in the row - exactly what the extension pass
    could not fix. Every strip becomes a full page on the row's pitch."""
    row1 = [
        (1920, 3790, 880, 2210), (4220, 4070, 800, 1900),
        (10640, 3870, 810, 2020), (17150, 3950, 1240, 1860),
        (19330, 4210, 1000, 1580), (19510, 3480, 730, 850),
        (21490, 4260, 1350, 1500), (25860, 4610, 1950, 1100),
    ]
    # context: a healthy row below fixes pitch and page size
    row2 = [(2010 + i * 2180, 6710, 2040, 2790) for i in range(12)]

    snapped, flags, notes, refused = snap_pages(row1 + row2, 2050, 2780)

    assert refused == [], notes
    assert len(snapped) == 7 + 12, snapped  # the two overlapping strips fuse
    for (x, y, w, h) in snapped:
        assert w == 2050 and h == 2780, (x, y, w, h)
    grown = [b for b, f in zip(snapped, flags) if f]
    assert len(grown) >= 7, "every strip page must be marked as snapped-grown"


def test_snap_leaves_healthy_pages_nearly_alone_and_unmarked():
    boxes = [(2010 + i * 2180, 6710, 2040, 2790) for i in range(6)]
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)

    assert refused == []
    assert not any(flags), "healthy pages must not be marked repaired"
    for (x, y, w, h), (sx, sy, sw, sh) in zip(sorted(boxes), sorted(snapped)):
        assert abs(x - sx) <= 30 and abs(y - sy) <= 30


def test_snap_fuses_stacked_fragments_and_vertical_strips_into_slots():
    """The old merge passes as special cases: anything x-overlapping within
    the row is one page slot."""
    boxes = [
        (2010, 6710, 2040, 2790),                       # whole
        (4190, 6680, 1350, 2760), (5540, 6680, 720, 2760),   # v-strips
        (6370, 6650, 1950, 1875), (6370, 8525, 1950, 885),   # h-fragments
    ]
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)

    assert refused == []
    assert len(snapped) == 3, snapped


def test_snap_refuses_a_detection_straddling_two_cells():
    """A detection bridging two pages' grid spans is the one geometry no
    page explains - exit 3 material, not silent repositioning. (A strip
    fully INSIDE a cell is fine wherever it sits - a washed page may keep
    only its middle.)"""
    boxes = [(2010, 6710, 2040, 2790), (2010 + 3 * 2180, 6710, 2040, 2790),
             (3400, 6790, 1400, 2100),   # spans the cell-0/cell-1 boundary
             # a second row pins the pitch
             (2010, 10500, 2040, 2790), (4190, 10500, 2040, 2790),
             (6370, 10500, 2040, 2790)]

    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused != [], (notes, snapped)
    assert any("bridges two pages" in nn for nn in notes), notes


def test_snap_reunites_the_field_cards_right_hand_strip():
    """Card 612130000029 detections 10+11+12 (real coordinates): 11 is the
    RIGHT strip of page 10's cell - gap-chaining would glue it to page 12.
    Cell assignment by center puts 10+11 together and leaves 12 whole."""
    boxes = [(2220 + i * 2180, 3190, 2040, 2780) for i in range(9)]
    boxes += [(21850, 3140, 760, 2780), (23010, 3130, 880, 2770),
              (24030, 3120, 2050, 2790)]
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)

    assert refused == [], notes
    assert len(snapped) == 11, snapped
    xs = sorted(b[0] for b in snapped)
    assert all(abs((b - a) - 2180) <= 40 for a, b in zip(xs, xs[1:])), xs


def test_snap_leaves_a_single_unsplittable_merge_alone():
    """One detection spanning two pages with no valley evidence: snapping
    would invent a split the binary cannot support - the merged-pages
    warning already covers it (C-contract from the merged-row e2e)."""
    boxes = [(2010, 6710, 2040, 2790), (4190, 6710, 4260, 2790)]
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused == []
    assert (4190, 6710, 4260, 2790) in snapped, snapped


def test_snap_splits_a_multi_member_fused_slot_on_the_pitch():
    """Healthy neighbours fused by the seam-gap tolerance re-emit as
    separate pages on the grid - fusion must never LOSE pages."""
    boxes = [(2010 + i * 2180, 6710, 2100, 2790) for i in range(4)]
    # 80px gaps (< the fusion tolerance) - the whole row is one slot
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused == []
    assert len(snapped) == 4, snapped
    xs = sorted(b[0] for b in snapped)
    assert all(abs((b - a) - 2180) <= 30 for a, b in zip(xs, xs[1:])), xs


def test_snap_single_strip_row_uses_pitch_from_other_rows():
    """A row holding only ONE narrow strip: phase comes from itself, pitch
    from the healthy rows, and the strip still becomes a full page -
    BOTTOM-anchored, because an anchor-less row has washed tops by
    definition (the 036 rule)."""
    boxes = [(2010 + i * 2180, 6710, 2040, 2790) for i in range(4)]
    boxes.append((4190, 10500, 700, 2100))
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused == []
    strip_page = [b for b in snapped if b[1] >= 9000]
    assert strip_page == [(4190, 12600 - 2780, 2050, 2780)], snapped


def test_snap_first_and_last_strip_of_a_row_get_full_pages():
    """Edge slots (no neighbour on one side) snap like interior ones."""
    boxes = [(2010, 6710, 600, 2100),                       # first: strip
             (4190, 6710, 2040, 2790), (6370, 6710, 2040, 2790),
             (8550, 6710, 500, 1800)]                       # last: strip
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused == []
    assert len(snapped) == 4
    for (x, y, w, h) in snapped:
        assert (w, h) == (2050, 2780)
    xs = sorted(b[0] for b in snapped)
    assert abs(xs[0] - 2010) <= 330 and abs(xs[-1] - 8550) <= 330, xs


def test_snap_card_where_everything_is_strips():
    """No trusted slot anywhere: phase falls back to the slots themselves.
    Positions stay near the detections; sizes are still the prior's."""
    boxes = [(2010 + i * 2180, 6710, 700, 2100) for i in range(4)]
    snapped, flags, notes, refused = snap_pages(boxes, 2050, 2780)
    assert refused == []
    assert all((b[2], b[3]) == (2050, 2780) for b in snapped)
    assert all(flags), "every strip page must be marked"


def test_page_size_tuning_stays_inside_the_prior_bound():
    """Detections at the very edge of the +-10% window tune the size to the
    window edge, never beyond it."""
    hi_w = int(PAGE_SIZE_PRIOR[0] * (1 + PAGE_SIZE_TOLERANCE)) - 1
    hi_h = int(PAGE_SIZE_PRIOR[1] * (1 + PAGE_SIZE_TOLERANCE)) - 1
    boxes = [(i * 2500, 100, hi_w, hi_h) for i in range(3)]
    pw, ph, note = resolve_page_size(boxes)
    assert note is None
    assert pw <= PAGE_SIZE_PRIOR[0] * (1 + PAGE_SIZE_TOLERANCE)
    assert ph <= PAGE_SIZE_PRIOR[1] * (1 + PAGE_SIZE_TOLERANCE)
    assert pw == hi_w and ph == hi_h


import re

FIELD_DATA = Path.home() / "Desktop" / "Mikrofiche-feltdata"


def _field_cards(report):
    """(stem, boxes, exit0, quality) per card in a field report folder."""
    folder = FIELD_DATA / report
    summary = (folder / "SAMMENDRAG.txt").read_text()
    cards = []
    for rapport_file in sorted(folder.glob("*_rapport.txt")):
        stem = rapport_file.name[:-len("_rapport.txt")]
        text = rapport_file.read_text()
        boxes = []
        in_block = False
        for line in text.splitlines():
            if "PAGE COORDINATES" in line:
                in_block = True
                continue
            if in_block:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 5 and parts[0].isdigit():
                    boxes.append(tuple(int(p) for p in parts[1:]))
                elif boxes:
                    break
        m = re.search(r"Card Quality: ([\d.]+)/100", text)
        status = next((l for l in summary.splitlines() if stem in l), "")
        if boxes and m:
            cards.append((stem, boxes, status.startswith("OK"),
                          float(m.group(1))))
    return cards


@pytest.mark.skipif(not FIELD_DATA.exists(), reason="field data not on disk")
def test_snap_regression_against_both_field_reports():
    """Every card from RAPPORT-2026-09-08-3 and -4 through the snap: no
    card may come out WORSE than today - no refusals on cards that passed,
    uniform page sizes, and the quality never drops. Card 612130000135
    (strips, quality 20.5 in -4) must come up measurably."""
    checked = 0
    for report in ("RAPPORT-2026-09-08-3", "RAPPORT-2026-09-08-4",
                   "RAPPORT-2026-09-08-5"):
        for stem, boxes, was_ok, q_before in _field_cards(report):
            pw, ph, note = resolve_page_size(boxes)
            assert note is None, (report, stem, "production card off-prior?")
            snapped, flags, notes, refused = snap_pages(boxes, pw, ph)
            if was_ok:
                assert refused == [], (report, stem, notes)
            refused_idx = {i for g in refused for i in g}
            exempt_like = [b for b in snapped
                           if (b[2], b[3]) != (pw, ph)]
            for b in exempt_like:
                assert (b[2] > 1.25 * pw or b[3] > 1.25 * ph
                        or refused), (report, stem, b)
            q_after = compute_card_quality(snapped, None)["total"]
            assert q_after >= q_before - 1, (report, stem, q_before, q_after)
            if stem == "612130000135_00012" and report.endswith("-4"):
                assert q_after > 50, (q_before, q_after)
            checked += 1
    assert checked >= 42, checked


# --- Coverage guard (mandatory, all modes) ----------------------------------
# Field card 612130000036 scored 100.0 with its entire first page row OUTSIDE
# every box (row-banding collapse dropped all cells onto row 2) - the worst
# "wrong that looks normal" so far. Foreground mass outside all page boxes
# must cap the score and warn loudly, in legacy and background-first alike.

from segment_microfiche import COVERAGE_WARN_SHARE, foreground_outside_boxes


def test_foreground_outside_boxes_measures_the_uncovered_share():
    b = np.zeros((100, 200), np.uint8)
    b[10:30, 10:90] = 255     # covered blob
    b[60:80, 10:90] = 255     # uncovered blob, same mass
    share = foreground_outside_boxes(b, [(5, 5, 100, 35)])
    assert abs(share - 0.5) < 0.02, share


def test_foreground_outside_boxes_is_zero_when_boxes_cover_all():
    b = np.zeros((100, 200), np.uint8)
    b[10:30, 10:90] = 255
    assert foreground_outside_boxes(b, [(0, 0, 200, 100)]) == 0.0
    assert foreground_outside_boxes(np.zeros((50, 50), np.uint8), []) == 0.0


def test_uncovered_foreground_caps_the_quality_score(tmp_path):
    """A card where a whole page row ends up outside the boxes must never
    score GOOD. Forced here with --no-split and a fused undetectable row?
    No - simplest honest reproduction: pages exist in two rows but boxes
    only cover one (we drive main with a fixture whose second row is all
    small rests below min size, so detection sees row 1 only while the
    binary holds row 2's mass)."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.full((3000, 6400), 180, 'uint8')
    for c in range(12):
        a[350:1050, 100 + c * 520:500 + c * 520] = 110       # row 1: real
    for c in range(12):
        a[1600:2300, 100 + c * 520:500 + c * 520] = 112      # row 2: real!
    # row 2 becomes vertical bars: wide enough to survive the detect-scale
    # erosion (200px -> 8px left), narrow enough to fail the min-size
    # filter - mass in the binary, but no boxes
    a[1600:2300, :] = 180
    for c in range(12):
        x = 100 + c * 520
        a[1600:2300, x + 100:x + 300] = 110
    pyvips.Image.new_from_memory(a.tobytes(), 6400, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "outside every page box" in (proc.stdout + proc.stderr).lower(), \
        proc.stdout
    m = re.search(r"Card Quality: ([\d.]+)/100", proc.stdout)
    assert float(m.group(1)) < 80, "uncovered content must cost the score"


# --- Robust row building (root causes 036 and 111, field 2026-09-08-5) ------
# 036: washed row-1 detections have LOW tops, so row 2 fell inside the
# span-limited top band -> ONE band, every cell y-anchored at row 2, twelve
# boxes with a whole page row uncovered (at quality 100). 111: bottom
# fragments formed their own band -> two overlapping "rows" of full pages.
# Rows are now built by transitive y-OVERLAP clustering: same-row members
# overlap each other (fragments overlap their anchors), different rows do
# not overlap at all.


def test_snap_keeps_a_washed_row_separate_from_the_row_below():
    """The 036 mechanism: row-1 detections are bottom-heavy (tops washed),
    row-2 tops lie within one page height of row-1's first top. Two rows
    must come out - with row 1 bottom-anchored ABOVE row 2, no overlap."""
    row1 = [(2340 + k * 2170, 4400, 2050, 1700) for k in range(6)]  # washed
    row2 = [(2340 + k * 2170, 6700, 2050, 2780) for k in range(4)]  # whole
    snapped, flags, notes, refused = snap_pages(row1 + row2, 2050, 2780)

    assert refused == [], notes
    assert len(snapped) == 10, snapped
    ys = sorted({b[1] for b in snapped})
    assert len(ys) == 2, f"row-banding collapse: {ys}"
    assert ys[1] - ys[0] >= 2000, f"rows overlap: {ys}"
    assert ys[0] + 2780 <= ys[1] + 100, f"row 1 overlaps row 2: {ys}"


def test_snap_folds_bottom_fragments_into_their_own_row():
    """The 111 mechanism: bottom fragments (top ~half a page below the row
    top) must join their row's cells - not form a second row of full pages
    stacked on the first."""
    fulls = [(2020 + k * 2180, 3295, 2040, 2780) for k in range(6)]
    frags = [(2225 + k * 2180, 4690, 1800, 1300) for k in range(6)]
    snapped, flags, notes, refused = snap_pages(fulls + frags, 2050, 2780)

    assert refused == [], notes
    assert len(snapped) == 6, ("fragments must merge into their pages, "
                               "not stack a second row", snapped)


def test_first_page_row_survives_the_header_mask(tmp_path):
    """A card whose first row crosses the 8% header band: masking cuts the
    detections' tops, but bottoms survive - the snap must give row 1 full
    pages again (extraction crops from the unmasked original)."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.full((3000, 6400), 180, 'uint8')
    for r in range(2):
        for c in range(12):
            y = 100 + r * 900          # row 1 starts INSIDE the 8% band
            a[y:y + 700, 100 + c * 520:500 + c * 520] = 110
    pyvips.Image.new_from_memory(a.tobytes(), 6400, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows_csv = [l for l in (out / "page_coordinates.csv").read_text().splitlines()[2:] if l]
    assert len(rows_csv) == 24, rows_csv
    tops = sorted({int(l.split(",")[2]) for l in rows_csv})
    heights = {int(l.split(",")[4]) for l in rows_csv}
    assert min(tops) <= 250, f"first row's top was eaten: {tops}"
    assert max(heights) >= 630, heights


# --- Position witnesses (field card 098) ------------------------------------
# The left half of 098's row 2 vanished: small detection rests were killed
# by the min-size filter BEFORE the snap could see them. A small blob in an
# empty grid cell proves the cell holds a page. Witnesses never create
# rows (a specks-only card still exits 2) and never vote on phase, pitch
# or row edges - they only claim cells.


def test_detect_can_collect_sub_min_witnesses():
    from segment_microfiche import detect_page_boxes
    b = np.zeros((1500, 2000), np.uint8)
    b[200:540, 60:460] = 255      # a real page
    b[220:280, 600:660] = 255     # a small rest - fails min size
    boxes, contours, _, _, witnesses = detect_page_boxes(
        b, 0, 100, 100, collect_witnesses=True)
    assert len(boxes) == 1
    assert len(witnesses) == 1
    wx, wy, ww, wh = witnesses[0]
    assert 550 <= wx <= 660 and ww < 100


def test_snap_gives_a_witnessed_empty_cell_a_page():
    """The 098 mechanism: an anchored row with an empty cell, plus a tiny
    rest inside that cell - the cell gets a full page, marked and logged."""
    boxes = [(2010, 6710, 2040, 2790), (6370, 6710, 2040, 2790),
             (8550, 6710, 2040, 2790)]
    witnesses = [(4600, 7400, 250, 300)]   # inside the gap cell at 4190
    snapped, flags, notes, refused = snap_pages(
        boxes, 2050, 2780, witnesses=witnesses)

    assert refused == []
    assert len(snapped) == 4, snapped
    witness_page = [b for b, f in zip(snapped, flags) if f]
    assert len(witness_page) == 1, snapped
    x, y, w, h = witness_page[0]
    assert (x, w, h) == (4190, 2050, 2780) and abs(y - 6710) <= 30, snapped
    assert any("witness" in n for n in notes), notes


def test_witnesses_never_create_rows_or_pages_outside_rows():
    boxes = [(2010, 6710, 2040, 2790), (4190, 6710, 2040, 2790)]
    witnesses = [(3000, 12000, 200, 200)]   # far below any row
    snapped, flags, notes, refused = snap_pages(
        boxes, 2050, 2780, witnesses=witnesses)
    assert len(snapped) == 2, snapped


def test_witnesses_in_occupied_cells_change_nothing():
    boxes = [(2010, 6710, 2040, 2790), (4190, 6710, 2040, 2790)]
    witnesses = [(2500, 7000, 200, 200)]
    snapped, flags, notes, refused = snap_pages(
        boxes, 2050, 2780, witnesses=witnesses)
    assert len(snapped) == 2 and not any(flags), (snapped, flags)


def test_half_a_row_of_rests_becomes_pages_end_to_end(tmp_path):
    """The 098 scenario: right half of a row detects normally, left half
    leaves only small rests. Every cell must come out as a page."""
    src = tmp_path / "612130000012_00016.jpg"
    a = np.full((3000, 6400), 180, 'uint8')
    for c in range(12):
        x = 100 + c * 520
        if c < 6:
            a[350:1050, x + 100:x + 300] = 110   # rests: survive erosion,
        else:                                     # fail min size
            a[350:1050, x:x + 400] = 110          # whole pages
    pyvips.Image.new_from_memory(a.tobytes(), 6400, 3000, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"

    proc = run_segmenter("-i", str(src), "-O", str(out), "--skip-extraction")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows_csv = [l for l in (out / "page_coordinates.csv").read_text().splitlines()[2:] if l]
    assert len(rows_csv) == 12, (rows_csv, proc.stdout)
    assert "position witness" in proc.stdout, proc.stdout


# --- Background-first binarization (--background-first, flagged) ------------
# Trond's Photoshop principle: the jacket is the only stable class - select
# the BACKGROUND and invert. Foreground = deviation from the local jacket
# level in EITHER direction, so faded, washed and half-dark content all
# count. Behind a flag until field-validated A/B against legacy on
# m4-studio. Calibration: band ratio 0.22 of local level - the blank fasit
# jacket (texture and all) stays within ~0.25, real content sits at 0.35+,
# and the faded-page fasit (28% darker than jacket, INVISIBLE to the global
# Otsu which lands below it) is caught from 0.18 up.


def _faded_card_file(tmp_path):
    """The 135 mechanism on the blank fasit: one clearly visible faded page
    that the global threshold cannot see."""
    img = pyvips.Image.new_from_file(
        str(REPO / "testdata" / "real_card_blank_10pct.jpg")).colourspace('b-w')
    a = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                   shape=[img.height, img.width]).astype(np.float32)
    a[700:1000, 500:710] *= 0.72
    src = tmp_path / "612130000012_00016.jpg"
    pyvips.Image.new_from_memory(a.astype(np.uint8).tobytes(), img.width,
                                 img.height, 1, 'uchar').write_to_file(str(src))
    return src


def test_background_first_finds_the_faded_page_legacy_misses(tmp_path):
    src = _faded_card_file(tmp_path)

    legacy = run_segmenter("-i", str(src), "-O", str(tmp_path / "card_legacy"),
                           "--skip-extraction")
    assert legacy.returncode == 2, ("fixture drift: legacy suddenly sees "
                                    "the faded page", legacy.stdout)

    bg = run_segmenter("-i", str(src), "-O", str(tmp_path / "card_bg"),
                       "--skip-extraction", "--background-first")
    assert bg.returncode == 0, bg.stdout + bg.stderr
    rows_csv = [l for l in (tmp_path / "card_bg" / "page_coordinates.csv")
                .read_text().splitlines()[2:] if l]
    assert len(rows_csv) >= 1, bg.stdout


def test_background_first_on_the_blank_fasit_still_exits_2(tmp_path):
    """The blank jacket is ALL background: nothing may survive - header
    remnants and stripe edges must die in the structure/band filters."""
    src = tmp_path / "612130000012_00016.jpg"
    shutil.copyfile(REPO / "testdata" / "real_card_blank_10pct.jpg", src)
    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction", "--background-first")
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_background_first_matches_legacy_on_a_healthy_card(tmp_path):
    src = tmp_path / "612130000012_00016.jpg"
    make_journal_card(src)
    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction", "--background-first")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Found 36 potential pages" in proc.stdout, proc.stdout


def test_background_first_finds_pages_pasted_on_the_blank_fasit(tmp_path):
    """Component 5's synthetic substrate: blank jacket + real page regions
    pasted at known cells = a position fasit for background-first."""
    blank = pyvips.Image.new_from_file(
        str(REPO / "testdata" / "real_card_blank_10pct.jpg")).colourspace('b-w')
    real = pyvips.Image.new_from_file(
        str(REPO / "testdata" / "real_card_10pct.jpg")).colourspace('b-w')
    a = np.ndarray(buffer=blank.write_to_memory(), dtype=np.uint8,
                   shape=[blank.height, blank.width]).copy()
    r = np.ndarray(buffer=real.write_to_memory(), dtype=np.uint8,
                   shape=[real.height, real.width])
    page = r[250:550, 2170:2380]          # one real (tape) page
    for x in (500, 1200, 1900):
        a[650:950, x:x + 210] = page
    src = tmp_path / "612130000012_00016.jpg"
    pyvips.Image.new_from_memory(a.tobytes(), blank.width, blank.height,
                                 1, 'uchar').write_to_file(str(src))

    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction", "--background-first")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows_csv = [l for l in (tmp_path / "card" / "page_coordinates.csv")
                .read_text().splitlines()[2:] if l]
    assert len(rows_csv) == 3, (rows_csv, proc.stdout)
    xs = sorted(int(l.split(",")[1]) for l in rows_csv)
    for got, want in zip(xs, (500, 1200, 1900)):
        assert abs(got - want) <= 120, (xs, proc.stdout)


def test_background_first_splits_touching_pages(tmp_path):
    """The deviation rule must flow into the split pass: two touching pages
    form one blob, and the valley between them only reads as background
    with the same background-first logic applied to the crop."""
    src = tmp_path / "612130000012_00016.jpg"
    h, w = 1500, 6400
    a = np.full((h, w), 180, 'float32')
    for c in range(12):
        x = 100 + c * 520
        if c == 5:
            x -= 60   # page 6 slides against page 5: gap 60 -> touching-ish
        a[200:900, x:x + 400] = 110
    pyvips.Image.new_from_memory(a.astype('uint8').tobytes(), w, h, 1,
                                 'uchar').write_to_file(str(src))

    proc = run_segmenter("-i", str(src), "-O", str(tmp_path / "card"),
                         "--skip-extraction", "--background-first")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows_csv = [l for l in (tmp_path / "card" / "page_coordinates.csv")
                .read_text().splitlines()[2:] if l]
    assert len(rows_csv) == 12, (rows_csv, proc.stdout)


def test_rapport_passes_extra_flags_through(tmp_path, monkeypatch):
    calls = []
    real_run = rapport.subprocess.run

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return real_run(cmd, **kw)

    monkeypatch.setattr(rapport.subprocess, "run", fake_run)
    src = tmp_path / "arkiv"
    src.mkdir()
    make_card(src / "612130000012_00001.jpg")
    rapport.run_report(src, tmp_path / "R", open_finder=False,
                       extra_args=["--background-first"])
    seg_calls = [c for c in calls if any("segment_microfiche" in str(p) for p in c)]
    assert seg_calls and "--background-first" in seg_calls[0]


def test_a_speck_is_not_a_position_witness():
    """Measured on the real fasit: a 20x10 rest of dirt claimed a phantom
    page cell. Witnesses need real mass (genuine rests are 1.5-2.5% of a
    page)."""
    boxes = [(2010, 6710, 2040, 2790), (6370, 6710, 2040, 2790),
             (8550, 6710, 2040, 2790)]
    snapped, flags, notes, refused = snap_pages(
        boxes, 2050, 2780, witnesses=[(4600, 7400, 20, 10)])
    assert len(snapped) == 3, snapped
