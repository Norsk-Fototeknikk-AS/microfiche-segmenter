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
    assert MAX_ROWS == 5
    assert MAX_PAGES_PER_ROW == 11


def test_too_many_pages_in_a_row_warns(tmp_path):
    """make_card's fixed pitch cannot hold 12 columns, so build a wide card."""
    src = tmp_path / "612130000012_00012.jpg"
    a = np.zeros((800, 6400), 'uint8')
    for c in range(12):
        a[200:540, 60 + c * 520:460 + c * 520] = 255
    pyvips.Image.new_from_memory(a.tobytes(), 6400, 800, 1, 'uchar').write_to_file(str(src))
    out = tmp_path / "card"
    proc = run_segmenter("-i", str(src), "-O", str(out),
                         "--skip-extraction", "--no-invert")
    assert proc.returncode == 0, proc.stderr
    assert "12 pages in one row" in proc.stdout, proc.stdout


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
