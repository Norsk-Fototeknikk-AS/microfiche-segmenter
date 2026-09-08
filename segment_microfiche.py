#!/usr/bin/env python3
"""
Microfiche page segmentation using Otsu thresholding.
Converts gigapixel JPG to 1-bit TIFF, finds page bounding boxes.
Reading order, --order:
  - 'rows' (default): top-to-bottom rows, left-to-right within each row
  - 'columns': left-to-right columns, top-to-bottom within each column
"""

import pyvips
import cv2
import numpy as np
from pathlib import Path
import argparse
import platform
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple
import os
import shutil
import sys

# === CONFIGURATION ===

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

# Physical card limits (Trond, 2026-09-04; per-row raised 2026-09-08 after
# rows of 12 were measured in production - 13 gives margin): at most 5 rows,
# and no column structure at all - rows start where they start. Exceeding
# either axis is itself a misdetection signal.
MAX_ROWS = 5
MAX_PAGES_PER_ROW = 13

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
# Detections that look like one page cut horizontally in two (stitching seam
# suspected). Content may be MISSING in the gap, so auto-merging is wrong and
# the card fails loudly instead of archiving half-pages as success.
EXIT_SUSPECT_FRAGMENTS = 3

# Fragment-pair signature: two detections covering the same x-span (interval
# IoU), separated by a small vertical gap, whose union is one page tall.
FRAGMENT_X_IOU = 0.8
FRAGMENT_MAX_GAP_RATIO = 0.15   # gap vs expected page height
# The union band is what separates a split page (union ~1x expected) from two
# whole pages in adjacent rows (union ~2x expected, must stay outside). The
# upper bound is generous because a card where EVERY page is split has no
# whole page left to anchor the expected height - it lands low, and the real
# pair must still fit under the bound (measured 1.6x on the seamed fasit).
FRAGMENT_UNION_MIN = 0.8
FRAGMENT_UNION_MAX = 1.8
FRAGMENT_MARK_COLOR = (0, 165, 255)  # orange boxes in both visualizations

# Phase 2, geometric completion (Trond, 2026-09-08): stitching is fixed and
# content is INTACT, so grid-matching fragment chains are MERGED into one
# page (crops are cut from the original graytone; a washed-out patch may
# still hold readable traces for OCR) and lone short detections are extended
# to their row's height. What does not reconcile with the grid still exits 3.
# Field calibration (28 production groups, 13 cards): every group tiled its
# union exactly (invented ~0); the worst card needed 28% of its pages
# repaired - thresholds sit well above both with margin.
GEOMETRY_MAX_INVENTED_SHARE = 0.3   # per MERGED page; extensions are exempt
                                    # (empty film is harmless, fabricated
                                    # content between fragments is not).
                                    # The group criteria already bound
                                    # invention near this - the cap is the
                                    # backstop if they are ever loosened.
GEOMETRY_MAX_REPAIR_SHARE = 0.5     # of the card's pages, else the card is
                                    # genuinely sick
EXTEND_MAX_WIDTH_RATIO = 1.25   # steg 10B: wider than this is not one page
GEOMETRY_SHORT_RATIO = 0.7          # below this share of row height = short
GEOMETRY_FULL_RATIO = 0.8           # at least this share = a full anchor
GEOMETRY_MARK_COLOR = (255, 80, 0)  # blue boxes for geometry-completed pages

# Vertical stripe merging (Trond's override, 2026-09-08): a page split into
# full-height STRIPS has the outline of one page, and the format guarantees
# uniform page sizes - same safety as the horizontal case. The union band is
# TIGHTER than the horizontal one: two real neighbour pages union to ~2x the
# page width plus a real gap (field pitch 2180 vs width 2040), so 1.2x
# excludes them with a wide margin - that exclusion is the whole risk.
STRIPE_UNION_MIN = 0.8
STRIPE_UNION_MAX = 1.2

# A card is pages on visible card background, so foreground can never be
# ~everything. Above this share the threshold split is meaningless (blank or
# washed-out scan) and the run fails loudly instead of emitting one giant
# "page". Real cards run well below this, even with narrow gaps.
FOREGROUND_SANE_MAX = 0.97

# Illumination flattening (production 2026-09-08: mottled panoramas after a
# machine upgrade - patchy brightness, content intact - put patches on the
# wrong side of the global Otsu threshold and silently ate pages). The
# low-frequency field is estimated per cell as a HIGH percentile: that tracks
# the bright class (jacket on journal cards, pages on Yamaha type), which
# mottling scales along with everything else, while page content does not
# read as lighting. Cell count is relative to width so a cell (~2400 px on a
# real panorama) stays coarser than a page and cannot hollow page interiors.
ILLUM_FIELD_CELLS = 12
# The planning thumbnail needs page/jacket structure RESOLVED, so its size is
# a target width, not a fixed scale - at a fixed 1% a test-sized card is a
# 64px mush where the field reads noise. 600px keeps real-panorama pages
# ~44px wide with visible jacket between rows.
ILLUM_THUMB_WIDTH = 600
ILLUM_FIELD_P = 90
ILLUM_FIELD_BLUR_SIGMA = 1.0   # in cells
ILLUM_FIELD_FLOOR = 0.4        # of field max: the dark surround around the
                               # card must not be boosted into fake foreground
# Flattening a flat image is ~identity, so the share of thumbnail pixels the
# flattening RE-CLASSIFIES is the mottle detector (field ratio is not - the
# dark surround dominates it even on healthy cards, measured 1.9 on the clean
# fasit vs 2.1 mottled). Measured: clean fasit cards 0.15-0.19%, the fasit
# with a synthetic blotch over the pages 0.93% - warn between, with margin
# both ways.
ILLUM_WARN_SHARE = 0.03

# Background-first binarization (--background-first, 2026-09-09, flagged
# until A/B-validated in production): the jacket is the only STABLE class -
# content varies wildly (faded, washed, half-dark), so select background
# and take the complement. Foreground = |pixel - local jacket level| above
# a relative band; deviation in EITHER direction counts, which is what
# catches faded pages sitting between the jacket and a global threshold.
# Mechanism choice (documented per Trond's component 5): a STATISTICAL band
# around the per-card p90 level field, not a direct diff against the blank
# card - jackets vary physically card to card (brown/gray stripes), so the
# blank card serves as CALIBRATION fasit instead: its jacket (texture and
# all) stays within ~0.25 of the local level, real content sits at 0.35+,
# and the faded-page fasit (28% darker than jacket, invisible to global
# Otsu) is caught from 0.18 up. Hence 0.22.
BG_BAND_RATIO = 0.22


def estimate_illumination_field(thumb):
    """Low-frequency illumination field from a small grayscale thumbnail,
    as a float32 grid of ILLUM_FIELD_CELLS across."""
    h, w = thumb.shape
    cw = ILLUM_FIELD_CELLS
    ch = max(3, round(cw * h / w))
    ys = np.linspace(0, h, ch + 1).astype(int)
    xs = np.linspace(0, w, cw + 1).astype(int)
    field = np.empty((ch, cw), np.float32)
    for i in range(ch):
        for j in range(cw):
            field[i, j] = np.percentile(
                thumb[ys[i]:ys[i + 1], xs[j]:xs[j + 1]], ILLUM_FIELD_P)
    field = cv2.GaussianBlur(field, (0, 0), ILLUM_FIELD_BLUR_SIGMA)
    return np.maximum(field, max(1.0, ILLUM_FIELD_FLOOR * float(field.max())))


def illumination_plan(thumb):
    """(field, norm, otsu_thresh, reclassified_share) from a small thumbnail.

    Every threshold downstream derives from otsu * field / norm: the full-res
    binary via a threshold SURFACE (preserving the threshold-first-then-resize
    order the detect pass depends on), the crop passes via local scalars - so
    all passes see the same flattened view without ever materializing a
    flattened gigapixel image.
    """
    field = estimate_illumination_field(thumb)
    norm = float(np.median(field))
    field_up = cv2.resize(field, (thumb.shape[1], thumb.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    flat = np.clip(thumb.astype(np.float32) * (norm / field_up),
                   0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(flat, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_thresh, _ = cv2.threshold(thumb, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    share = float(np.mean((flat >= thresh) != (thumb >= raw_thresh)))
    return field, norm, float(thresh), share


def otsu_excluding(thumb, mask):
    """Otsu over the pixels where mask == 0, or None if nothing is left.

    Step two of the staircase (C9, steg 7). A page sits next to the FRAME in
    grey level, not next to the jacket - fasit levels frame 2, page 33,
    stripe 55, jacket 226 - so a healthy card has one dark cluster and Otsu
    lands in the wide gap below the jacket. When over-exposure lifts the
    pages toward the jacket, the frame is the only dark mass left and Otsu
    splits FRAME against everything else (measured 85-111 on six field
    cards), putting every page on the background side. Take the frame and
    the known structure out of the histogram and the false valley goes with
    it. Masking is by POSITION, never by level, so the frame's colour -
    which varies from jacket to jacket - does not matter.
    """
    keep = thumb[mask == 0]
    if keep.size < 2 or keep.min() == keep.max():
        return None
    thresh, _ = cv2.threshold(keep.reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh)


# The staircase's trigger and its proof (C9, steg 7). Validated on all 88
# production cards: border share is a FRACTION of total foreground, so a
# card with few pages reads high even when thresholding is perfect (card
# 418: 5 pages, 85.5 %, quality 100). It therefore cannot trigger on its
# own - the trigger is that the first pass demonstrably did not find the
# card, AND that what foreground it did find is essentially all frame.
STEP2_BORDER_TRIGGER = 0.40      # lowest triggering field card: 62.4 %
# Kept for the log only: the border share is reported before and after,
# but it does not decide (steg 9C - it punished small cards).


def frame_mask(thumb, thresh, header_px):
    """Pixels step two keeps OUT of its histogram: the header band and the
    dark structure connected to the image border - frame, edge bands and the
    stripes that reach the edge. Selected by POSITION, never by level, so
    the frame's colour (which varies from jacket to jacket) does not matter.
    """
    dark = ((thumb <= thresh) * 255).astype(np.uint8)
    h, w = dark.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        for y in (0, h - 1):
            if dark[y, x]:
                cv2.floodFill(dark, ff, (x, y), 0)
    for y in range(h):
        for x in (0, w - 1):
            if dark[y, x]:
                cv2.floodFill(dark, ff, (x, y), 0)
    mask = ((thumb <= thresh) & (dark == 0)).astype(np.uint8)
    if header_px > 0:
        mask[:header_px, :] = 1
    return mask


def illumination_local_threshold(otsu_thresh, field, norm, box,
                                 full_w, full_h):
    """Scalar threshold for a full-res crop: otsu scaled by the mean field
    over the box. Exactly equivalent to flattening where the field is locally
    constant - and the field is smoother than any single detection."""
    fh, fw = field.shape
    x, y, w, h = box
    x0 = min(fw - 1, max(0, int(x * fw / full_w)))
    y0 = min(fh - 1, max(0, int(y * fh / full_h)))
    x1 = max(x0 + 1, min(fw, -(-(x + w) * fw // full_w)))
    y1 = max(y0 + 1, min(fh, -(-(y + h) * fh // full_h)))
    return otsu_thresh * float(field[y0:y1, x0:x1].mean()) / norm


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


# A detection this many times wider/taller than the median page, while the
# other dimension is a thin SLIVER of the median, is a stitch band - not a
# page. Deliberately a SHAPE test: size alone says nothing, because real
# journals hold pages of genuinely different sizes.
BAND_RATIO = 2.5

# The sliver criterion (2026-09-03): weak page edges can fuse a whole ROW into
# one detection - several pages wide but FULL page height. Width cannot tell a
# merged row from a band (a full-width band and a fully merged row are equally
# wide); thinness can. The documented real band was 0.44x the median height; no
# real page is anywhere near that flat. Above this the box is merged content -
# kept, never dropped, until splitting exists.
BAND_MAX_THICKNESS = 0.75


def drop_band_detections(boxes, contours, band_ratio):
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
        wide_band = (w > median_w * band_ratio
                     and h <= median_h * BAND_MAX_THICKNESS)
        tall_band = (h > median_h * band_ratio
                     and w <= median_w * BAND_MAX_THICKNESS)
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


# A row of the binary counts as "full width" above this foreground share.
# Measured on the first real journal card (2026-09-04): structure rows (stripes,
# header/bottom bands) run 0.86-1.0 coverage, rows holding separated pages ~0.2.
# It stays 0.85 as the CANDIDATE threshold (steg 4A): a thin real stripe can
# fall below a higher bar at its edges. What a candidate must then prove is
# below.
STRIPE_COVERAGE = 0.85

# Stripe geometry, measured full-res on the 21505 px tall production card -
# all 16 field cards in BOTH A/B modes 2026-09-08, plus the committed fasit.
# Kept as ratios of image height so the same numbers hold at detect scale.
# The jacket is constant: every card carries seven structure runs - a top
# band, five stripes and a bottom band - on a per-card raster of pitch
# 3360-3470. A row of 12 pages covers 84.6 % of the width, which is why a
# coverage dip inside a row used to be deleted as "structure" (card 111
# lost the bottom of row 2 that way; 036, 050, 098, 104 and 029 the same).
STRIPE_REF_HEIGHT = 21505.0
PAGE_SIZE_PRIOR_H = 2780        # page height of the journal format (C15)
STRIPE_MIN_H = 80 / STRIPE_REF_HEIGHT        # real stripes 100-420 px; the
                                             # false slivers 10-20 px
STRIPE_MERGE_GAP = 60 / STRIPE_REF_HEIGHT    # card 074's stripe is cut in
                                             # two 40 px apart; real stripes
                                             # sit 3400 px apart
STRIPE_RASTER_TOL = 150 / STRIPE_REF_HEIGHT  # false runs sit 400-600 px off
# Solidity floor. First set to 0.95 from the fasit cards alone (their
# stripes measure 0.99-1.00 against 0.846 for a 12-page row). The A/B run of
# e18e143 logged coverage per run in the field for the first time and showed
# 0.95 was too high: 17 REAL stripes on cards 135 and 142 measure 0.92-0.95
# at 100-160 px and were refused, which cost those cards every row slot they
# had (142 standard ended with one structure run, 135 with two) - the cards
# still passed because clear_border_connected removes what survives here.
# False runs measure 0.85-0.89 across all 32 card runs, highest 0.89, so the
# floor belongs in that gap. Replayed over both modes: 0.95 leaves 28 of 32
# cards with their seven structure runs, 0.92 and below leaves 32 of 32.
# The raster is the deciding test regardless - a solid run in the wrong
# place is page content.
STRIPE_SOLID_COVERAGE = 0.90
# A stripe raster is one page row plus a stripe: the field pitch is
# 3360-3470 against a 2780 px page, i.e. 1.21-1.25 page heights. Capping at
# 1.6 makes DOUBLE pitch (2.5x) impossible - without the cap, a card with
# one missing stripe hands the fit to a raster of every other stripe, which
# fills perfectly (3 of 3) and beats the real one (4 of 5), leaving the
# stripes between it in the binary and dissolving a row slot.
STRIPE_MAX_PITCH = 1.6 * PAGE_SIZE_PRIOR_H / STRIPE_REF_HEIGHT


def coalesce_runs(runs, max_gap):
    """Merge runs separated by at most max_gap. A stripe cut in two by noise
    is one stripe (card 074: 6100-6110 + 6150-6400)."""
    merged = []
    for a, b in sorted(runs):
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def fit_stripe_raster(centers, min_pitch, tol, max_pitch=None):
    """(anchor, pitch) of the jacket's stripe raster, or None when there is
    too little evidence to fit one. Fitted per card, not assumed: the first
    stripe measures 5820-6370 across the field cards.

    Pitch is bounded above (max_pitch): a raster of every OTHER stripe fills
    perfectly on a card with one stripe missing, and would win on
    completeness alone.

    Scored on how COMPLETELY the raster is filled, then on how many
    candidates it explains. Support alone is not enough - a dense cluster of
    false runs inside one row (card 098 had eight) supports a finer pitch
    that hits more candidates while leaving most of its own positions empty.
    The jacket's stripes are periodic AND complete, so the raster that
    fills every position between its first and last hit is the real one.
    """
    best = ((0.0, 0), None, None)
    if len(centers) >= 3:
        for anchor in centers:
            for other in centers:
                pitch = other - anchor
                if pitch < min_pitch or (max_pitch and pitch > max_pitch):
                    continue
                hits = [c for c in centers
                        if min((c - anchor) % pitch,
                               pitch - (c - anchor) % pitch) <= tol]
                if len(hits) < 3:
                    continue
                first = round((min(hits) - anchor) / pitch)
                last = round((max(hits) - anchor) / pitch)
                positions = last - first + 1
                score = (len(hits) / positions, len(hits))
                if score > best[0]:
                    best = (score, anchor, pitch)
    return (best[1], best[2]) if best[0][1] >= 3 else None


def classify_structure_runs(runs, coverages, height, min_page_h,
                            top_boundary, report_scale=1.0):
    """Which full-width runs are card structure (row boundaries), and why the
    others are not.

    A run is structure only if it is the top band (starts at or above the
    header mask), the bottom band (reaches the image edge), or a stripe:
    thick enough, solid enough, and sitting on the card's stripe raster.
    Everything else is page content and must be left alone - deleting it is
    what ate rows in the field.

    Returns (structure_runs, rejected) with rejected as [(run, reason)].
    Numbers in the reasons are multiplied by report_scale so the log speaks
    full-res while the work happens at detect scale.
    A raster position with no candidate is reported the same way as a
    zero-length run, so a missing stripe is visible in the log; the slot
    then spans two rows and the snap invariants still hold.
    """
    min_h = STRIPE_MIN_H * height
    tol = STRIPE_RASTER_TOL * height
    R = report_scale
    structure, rejected, candidates = [], [], []
    for run, cov in zip(runs, coverages):
        a, b = run
        thick = b - a
        if a <= top_boundary:
            structure.append(run)
        elif b >= height:
            structure.append(run)
        elif thick >= min_page_h:
            rejected.append((run, f"kept run {a * R:.0f}-{b * R:.0f}: "
                                  f"{thick * R:.0f} px, coverage {cov:.2f} - "
                                  "page-height (a merged page row, not a "
                                  "stripe)"))
        elif thick < min_h:
            rejected.append((run, f"kept run {a * R:.0f}-{b * R:.0f}: "
                                  f"{thick * R:.0f} px, coverage {cov:.2f} - "
                                  f"too thin for a stripe (needs "
                                  f"{min_h * R:.0f} px)"))
        elif cov < STRIPE_SOLID_COVERAGE:
            rejected.append((run, f"kept run {a * R:.0f}-{b * R:.0f}: "
                                  f"{thick * R:.0f} px, coverage {cov:.2f} - "
                                  "not solid enough (a stripe measures "
                                  f"{STRIPE_SOLID_COVERAGE:.2f}+)"))
        else:
            candidates.append((run, cov))

    centers = [(a + b) / 2 for (a, b), _ in candidates]
    raster = fit_stripe_raster(centers, 2 * min_page_h, tol,
                               STRIPE_MAX_PITCH * height)
    if raster is None and candidates:
        rejected.append(((0, 0), f"raster not fitted: {len(candidates)} "
                                 "candidate(s) - all kept as structure"))
    for (run, cov), c in zip(candidates, centers):
        a, b = run
        if raster is None:
            structure.append(run)
            continue
        anchor, pitch = raster
        d = (c - anchor) % pitch
        off = min(d, pitch - d)
        if off <= tol:
            structure.append(run)
        else:
            rejected.append((run, f"kept run {a * R:.0f}-{b * R:.0f}: "
                                  f"{(b - a) * R:.0f} px, coverage {cov:.2f} "
                                  f"- off-raster by {off * R:.0f} px"))

    if raster is not None and candidates:
        anchor, pitch = raster
        on = sorted(c for c in centers
                    if min((c - anchor) % pitch,
                           pitch - (c - anchor) % pitch) <= tol)
        pos = anchor + round((on[0] - anchor) / pitch) * pitch
        while pos <= on[-1] + tol:
            if all(abs(c - pos) > tol for c in on):
                rejected.append(((int(pos), int(pos)),
                                 f"MISSING stripe at raster position "
                                 f"{pos * R:.0f} - its slot spans two rows"))
            pos += pitch

    return sorted(structure), rejected


def remove_structure_rows(binary_img, min_page_h, top_boundary,
                          report_scale=1.0):
    """Delete full-width row-runs that cannot be pages, in place.

    The light journal jackets have dark edge-to-edge stripes between rows and
    dark bands along the header and bottom. Connectivity cannot separate them
    from pages - a sleeve can physically overlap its stripe (seen on the real
    card) - but geometry can: structure runs are thinner than any possible
    page, or touch the image boundary / the header-mask line. A fully merged
    row of pages is full-width too, but page-HEIGHT and floating mid-card, so
    it survives (and is handled by the merged-pages warning downstream).

    Returns (number of runs deleted, [(y_start, y_end), ...] of those runs,
    [(run, reason)] for every candidate NOT deleted). The deleted runs are
    the card's row boundaries (steg 2, 2026-09-08): a page never crosses a
    stripe, so snap_pages uses them as row slots. Which runs are structure
    at all is decided by classify_structure_runs (steg 4A) - full width is
    not enough, because a row of 12 pages is 84.6 % wide.
    """
    h, w = binary_img.shape
    coverage = (binary_img > 0).sum(axis=1) / w

    # Raw runs of full-width rows...
    runs = []
    y = 0
    while y < h:
        if coverage[y] > STRIPE_COVERAGE:
            start = y
            while y < h and coverage[y] > STRIPE_COVERAGE:
                y += 1
            runs.append([start, y])
        else:
            y += 1

    # ...coalesced across gaps: noise can make a band's coverage straddle the
    # threshold row by row, shredding it into 1-row "stripes"; and a real
    # stripe can be cut in two (card 074, 40 px apart). Real stripes sit
    # thousands of rows apart, so bridging STRIPE_MERGE_GAP is safe.
    raw = [tuple(r) for r in runs]
    merged = coalesce_runs(raw, max(1, int(round(STRIPE_MERGE_GAP * h))))
    # Coverage over the rows that were actually full-width, NOT over the
    # bridged gaps - those sit below the bar by definition, and averaging
    # them in drags a cut stripe (card 074) toward the solidity floor.
    covs = []
    for a, b in merged:
        rows = [coverage[ra:rb] for ra, rb in raw if ra >= a and rb <= b]
        covs.append(float(np.concatenate(rows).mean()) if rows
                    else float(coverage[a:b].mean()))

    structure, rejected = classify_structure_runs(
        merged, covs, h, min_page_h, top_boundary, report_scale)
    by_run = dict(zip(merged, covs))
    notes = [(run, f"stripe {run[0] * report_scale:.0f}-"
                   f"{run[1] * report_scale:.0f}: "
                   f"{(run[1] - run[0]) * report_scale:.0f} px, "
                   f"coverage {by_run.get(run, 1.0):.2f}")
             for run in structure] + rejected
    for start, end in structure:
        binary_img[start:end, :] = 0
    return len(structure), [(int(a), int(b)) for a, b in structure], notes


def clear_border_connected(binary_img):
    """Zero out foreground connected to the image border, in place.

    The card's own structure - the dark frame and the edge-to-edge stripes
    between rows - always reaches the image border; a page never does. On the
    light journal cards (dark pages, --invert) that structure is one connected
    component ENCLOSING every page, so RETR_EXTERNAL would return only the
    frame and lose all pages inside it. Removing border-connected foreground
    "diffs away" the card itself and leaves only floating detections: pages.

    Returns the share of foreground pixels that were removed, for the log.
    """
    h, w = binary_img.shape
    before = np.count_nonzero(binary_img)
    if before == 0:
        return 0.0
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = ([(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)]
             + [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)])
    for x, y in seeds:
        if binary_img[y, x]:
            cv2.floodFill(binary_img, mask, (x, y), 0)
    return 1.0 - np.count_nonzero(binary_img) / before


def detect_page_boxes(binary_img, header_skip_px, min_w, min_h,
                      collect_witnesses=False, structure_rows_out=None,
                      report_scale=1.0, border_share_out=None):
    """The detect-scale pipeline: mask header, remove card structure, erode,
    drop border-connected foreground, find and size-filter contours.

    Mutates and returns binary_img (the processed view is what the
    visualization shows). Returns (boxes, contours, binary_img, log_lines);
    with collect_witnesses=True a fifth element holds the sub-min-size
    boxes - too small to be pages, but survivors of the erosion, so they
    witness that their grid cell holds SOMETHING (field card 612130000098:
    half a row of small rests was silently discarded here). A list passed
    as structure_rows_out receives the deleted stripe runs (detect-scale
    y-intervals) - the row slots for snap_pages.
    """
    log_lines = []
    if header_skip_px > 0:
        binary_img[:header_skip_px, :] = 0

    min_page_h = max(1, min_h)
    removed_rows, structure_runs, structure_notes = remove_structure_rows(
        binary_img, min_page_h, header_skip_px, report_scale)
    if structure_rows_out is not None:
        structure_rows_out.extend(structure_runs)
    # Thickness and coverage for the stripes we kept AND the reason for every
    # run we refused: the next A/B calibrates the thresholds against these.
    for _run, text in structure_notes:
        log_lines.append("  " + text)
    if removed_rows:
        log_lines.append(f"Removed {removed_rows} full-width structure "
                         "row-run(s) (stripes / edge bands)")

    kernel = np.ones((DETECT_ERODE_KERNEL, DETECT_ERODE_KERNEL), np.uint8)
    binary_img = cv2.erode(binary_img, kernel, iterations=DETECT_ERODE_ITERATIONS)

    removed = clear_border_connected(binary_img)
    if border_share_out is not None:
        border_share_out.append(removed)
    if removed > 0:
        log_lines.append(f"Removed border-connected structure: "
                         f"{removed:.1%} of foreground")

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    kept_contours = []
    witnesses = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= min_h:
            boxes.append((x, y, w, h))
            kept_contours.append(c)
        else:
            witnesses.append((x, y, w, h))
    if collect_witnesses:
        return boxes, kept_contours, binary_img, log_lines, witnesses
    return boxes, kept_contours, binary_img, log_lines


# No single microfiche page spans close to the whole card in either dimension
# (real cards run up to MAX_PAGES_PER_ROW pages per row, up to MAX_ROWS rows).
# A detection wider or taller than
# this share of the image is a polarity artifact - a full row or column read
# as foreground - not a page.
PAGE_MAX_SPAN = 0.6


def page_likeness_score(boxes, width, height):
    """Score a detection result: +1 per plausible page, -1 per impossible one.

    The right threshold polarity yields floating page-sized boxes; the wrong
    one yields full-width rows (or nothing). Counting boxes alone cannot tell
    those apart - six rows outnumber two pages - so impossible spans count
    AGAINST the polarity that produced them.
    """
    score = 0
    for _, _, w, h in boxes:
        if w > width * PAGE_MAX_SPAN or h > height * PAGE_MAX_SPAN:
            score -= 1
        else:
            score += 1
    return score


def autodetect_inversion(binary_img, header_skip_px, min_w, min_h):
    """Decide threshold polarity by trying both and scoring page-likeness.

    Border sampling cannot decide this: the dark mounting surround frames both
    card types, so the border ring reads as background either way (measured on
    the real journal card, 2026-09-04). Instead run the cheap detect-scale
    pipeline on both polarities and keep the one producing page-like boxes.
    A scoreless tie prefers inverted: the production default is the journal
    card type - dark pages on a light jacket (Trond, 2026-09-04). Either way
    a tie ends in the loud no-pages failure downstream.

    binary_img is not modified; trials run on copies.
    """
    H, W = binary_img.shape
    normal, _, _, _ = detect_page_boxes(
        binary_img.copy(), header_skip_px, min_w, min_h)
    inverted, _, _, _ = detect_page_boxes(
        cv2.bitwise_not(binary_img), header_skip_px, min_w, min_h)
    return page_likeness_score(inverted, W, H) >= page_likeness_score(normal, W, H)


# A projection valley must drop below this share of the box's median level to
# count as a gap between pages. Measured on the real card: gap 0.45 against
# page level 0.99 (ratio 0.45); in-page content variation stays far above.
SPLIT_VALLEY_RATIO = 0.6


def find_projection_valleys(share, min_gap):
    """Find gap positions in a 1-D foreground-share profile of a merged box.

    A valley is a run of at least min_gap positions whose share drops below
    SPLIT_VALLEY_RATIO x the profile's median, not touching either end (a low
    run at the edge is the box boundary, not an internal gap). Returns the
    center index of each valley.
    """
    share = np.asarray(share, dtype=float)
    threshold = float(np.median(share)) * SPLIT_VALLEY_RATIO
    below = share < threshold
    valleys = []
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            start = i
            while i < n and below[i]:
                i += 1
            if start > 0 and i < n and (i - start) >= min_gap:
                valleys.append((start + i - 1) // 2)
        else:
            i += 1
    return valleys


# Scan scale for the split pass: same as the local refine pass. The real
# valley (24 full-res px) is ~5px here - resolvable, where the 10% detect
# scale blurs it into the pages.
SPLIT_SCAN_SCALE = 0.2


def split_box_by_projection(input_file, box, otsu_thresh, invert, min_w,
                            min_h, deviation=None):
    """Split one detection into the pages it contains, by projection valleys.

    Loads the box region, downsamples to SPLIT_SCAN_SCALE and thresholds
    AFTER the resize: averaging first is what makes a dirty gap (mixed
    dark/light, like the real card's overlapping tapes at 45% foreground)
    read as background, while the full-res-thresholded detect pass read it
    as page and merged the neighbors.

    Column valleys split vertically, row valleys horizontally (a fused block
    splits into a grid). A split line that would leave a piece smaller than
    min_w/min_h (full-res) is noise, not a gap. Returns full-res boxes;
    [box] unchanged when no valley is found.
    """
    x, y, w, h = box
    img = pyvips.Image.new_from_file(input_file, access='random')
    region = img.crop(x, y, w, h)
    if region.bands > 1:
        region = region.colourspace('b-w')
    small = region.resize(SPLIT_SCAN_SCALE)
    a = np.ndarray(buffer=small.write_to_memory(), dtype=np.uint8,
                   shape=[small.height, small.width])
    if deviation is not None:
        # Background-first: content is any deviation from the local jacket
        # level - same rule as the main pass, scalar per crop.
        level, band = deviation
        fg = np.abs(a.astype(np.float32) - level) > band
    else:
        fg = a >= otsu_thresh
        if invert:
            fg = ~fg

    min_gap = max(2, int(10 * SPLIT_SCAN_SCALE))
    col_cuts = find_projection_valleys(fg.mean(axis=0), min_gap)
    row_cuts = find_projection_valleys(fg.mean(axis=1), min_gap)

    def segments(cuts, length, min_len):
        # A cut that would leave an under-sized piece is noise - typically an
        # edge artifact from the background rim inside the box (seen on the
        # real card: a col-7 artifact next to the real col-430 gap). Judge
        # each cut alone, so an artifact never vetoes a real gap.
        kept = []
        prev = 0
        for cut in (int(c / SPLIT_SCAN_SCALE) for c in cuts):
            if cut - prev >= min_len and length - cut >= min_len:
                kept.append(cut)
                prev = cut
        edges = [0] + kept + [length]
        return [(edges[i], edges[i + 1] - edges[i])
                for i in range(len(edges) - 1)]

    x_parts = segments(col_cuts, w, min_w)
    y_parts = segments(row_cuts, h, min_h)
    if len(x_parts) == 1 and len(y_parts) == 1:
        return [box]
    return [(x + px, y + py, pw, ph)
            for (py, ph) in y_parts for (px, pw) in x_parts]


def erosion_radius(kernel_size, iterations):
    """Pixels eaten from each side of a blob by cv2.erode with this kernel."""
    return (kernel_size // 2) * iterations


def expected_page_height(boxes):
    """Robust page height for a card: the tallest detection, capped at 1.5x
    the 75th percentile of heights.

    Pages on a card are near-uniform and fragments (split pages) are always
    SHORTER than a whole page, so the tallest box is a whole page even on a
    card where everything else is fragments - a badly seamed card can be half
    fragments, which sinks the median. The percentile cap keeps one outlier
    (an unsplittable vertical merge, ~2 pages tall) from dragging the
    estimate up to double height.
    """
    heights = [h for (_, _, _, h) in boxes]
    return float(min(max(heights), 1.5 * np.percentile(heights, 75)))


def _x_interval_iou(a, b):
    left = max(a[0], b[0])
    right = min(a[0] + a[2], b[0] + b[2])
    if right <= left:
        return 0.0
    union = max(a[0] + a[2], b[0] + b[2]) - min(a[0], b[0])
    return (right - left) / union


def find_fragment_groups(boxes, union_min=FRAGMENT_UNION_MIN,
                         union_max=FRAGMENT_UNION_MAX):
    """Index groups of detections that look like ONE page cut horizontally.

    A seam-damaged page can come back as 2, 3 or 4 stacked fragments
    (production 2026-09-07: stacks of 3-4 where no PAIR reaches the union
    band). So: link boxes sharing an x-span (interval IoU >= FRAGMENT_X_IOU)
    with a small vertical gap, take the transitive chains, and inside each
    chain (sorted by y) report every maximal contiguous window whose union
    height matches the expected page height. The union band is what separates
    split pages (~1x expected) from whole pages in adjacent rows (~2x) - and
    the windowing keeps a tight next-row neighbour from hiding a real stack
    by pushing the whole chain's union past the band.
    """
    n = len(boxes)
    if n < 2:
        return []
    exp_h = expected_page_height(boxes)
    max_gap = FRAGMENT_MAX_GAP_RATIO * exp_h

    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            top, bot = ((boxes[i], boxes[j]) if boxes[i][1] <= boxes[j][1]
                        else (boxes[j], boxes[i]))
            if _x_interval_iou(top, bot) < FRAGMENT_X_IOU:
                continue
            if bot[1] - (top[1] + top[3]) > max_gap:
                continue
            adj[i].add(j)
            adj[j].add(i)

    seen = set()
    groups = []
    for start in range(n):
        if start in seen or not adj[start]:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comp.sort(key=lambda k: boxes[k][1])

        # Greedy top-down: the longest contiguous window still inside the
        # union band becomes a group; scanning resumes after it.
        i = 0
        while i < len(comp) - 1:
            best_end = None
            for j in range(i + 1, len(comp)):
                window = comp[i:j + 1]
                top_y = min(boxes[k][1] for k in window)
                bot_y = max(boxes[k][1] + boxes[k][3] for k in window)
                union_h = bot_y - top_y
                if union_h > union_max * exp_h:
                    break
                if union_h >= union_min * exp_h:
                    best_end = j
            if best_end is None:
                i += 1
            else:
                groups.append(tuple(sorted(comp[i:best_end + 1])))
                i = best_end + 1
    return sorted(groups)


def find_stripe_groups(boxes):
    """Index groups of detections that look like ONE page cut vertically.

    Transposing x<->y and w<->h turns vertical strips into the horizontal
    stacking problem, so the whole chain machinery is reused: y-IoU >= 0.8,
    horizontal gap <= 15% of expected page width (which expected_page_height
    computes on the transposed boxes), union WIDTH in the tight stripe band.
    Two separate half-width documents in neighbouring frames never link -
    their frames put a real gap between them.
    """
    transposed = [(y, x, h, w) for (x, y, w, h) in boxes]
    return find_fragment_groups(transposed, union_min=STRIPE_UNION_MIN,
                                union_max=STRIPE_UNION_MAX)


def complete_geometry(boxes):
    """Phase 2: let the known sheet size override a defective binary.

    Fragment chains from find_fragment_groups (they already reconcile with
    the expected page height) are merged into their union box; a merge is
    refused when it would INVENT more than GEOMETRY_MAX_INVENTED_SHARE of
    the page area - that is fabrication, not repair. Then lone short
    detections in a row with at least two full-height anchors are extended
    to the row's top edge and height (pages share their top edge within a
    row; verified on the fasit card).

    Returns (new_boxes, repaired_flags, notes, refused_groups). The caller
    fails the card for refused groups and for over-repair - the guard is
    not weakened, it just gets a repair step in front of it.
    """
    n = len(boxes)
    if n < 2:
        return list(boxes), [False] * n, [], []
    exp_h = expected_page_height(boxes)

    notes = []
    refused = []
    consumed = set()
    merged = []
    # Lower union bound REMOVED for merging (phase 3, 2026-09-08): a chain
    # under 0.8x expected height is a page that LOST height to a defect
    # (field pair 17+27: union 0.79x) - it merges here and the extension
    # pass completes it to the row's anchors. The upper bound stays: it is
    # the cross-row guard. Row-boundary safety for the short results is the
    # gap criterion - field row gaps (~840-920px) are twice the allowed gap.
    for g in find_fragment_groups(boxes, union_min=0.0):
        parts = [boxes[i] for i in g]
        x0 = min(b[0] for b in parts)
        y0 = min(b[1] for b in parts)
        x1 = max(b[0] + b[2] for b in parts)
        y1 = max(b[1] + b[3] for b in parts)
        union_area = (x1 - x0) * (y1 - y0)
        covered = sum(b[2] * b[3] for b in parts)
        invented = max(0.0, 1.0 - covered / union_area)
        pages = "+".join(str(i + 1) for i in g)
        if invented > GEOMETRY_MAX_INVENTED_SHARE:
            refused.append(g)
            notes.append(f"REFUSED merge of detections {pages}: "
                         f"{invented:.0%} of the page would be invented")
            continue
        consumed.update(g)
        sub_band = (y1 - y0) < FRAGMENT_UNION_MIN * exp_h
        merged.append(((x0, y0, x1 - x0, y1 - y0), sub_band))
        notes.append(f"merged {len(g)} fragments (detections {pages}) into "
                     f"one page at ({x0}, {y0}), {invented:.0%} invented"
                     + (" (short union - completing below)" if sub_band
                        else ""))

    contested = {i for g in refused for i in g}
    entries = [[boxes[i], False, i in contested, False]
               for i in range(n) if i not in consumed]
    entries += [[b, True, False, sub] for b, sub in merged]

    # Vertical stripes, on the horizontally-repaired boxes (a page split
    # into quadrants heals fully: the two half-width columns from the merge
    # above unite here). Contested fragments stay out.
    stripe_groups = find_stripe_groups([e[0] for e in entries])
    stripe_consumed = set()
    stripe_entries = []
    for g in stripe_groups:
        if any(entries[i][2] for i in g):
            continue
        parts = [entries[i][0] for i in g]
        x0 = min(b[0] for b in parts)
        y0 = min(b[1] for b in parts)
        x1 = max(b[0] + b[2] for b in parts)
        y1 = max(b[1] + b[3] for b in parts)
        union_area = (x1 - x0) * (y1 - y0)
        invented = max(0.0, 1.0 - sum(b[2] * b[3] for b in parts) / union_area)
        dets = "+".join(str(i + 1) for i in g)
        if invented > GEOMETRY_MAX_INVENTED_SHARE:
            refused.append(g)
            notes.append(f"REFUSED merge of vertical stripes {dets}: "
                         f"{invented:.0%} of the page would be invented")
            continue
        stripe_consumed.update(g)
        stripe_entries.append([(x0, y0, x1 - x0, y1 - y0), True, False, False])
        notes.append(f"merged {len(g)} vertical stripes (detections {dets}) "
                     f"into one page at ({x0}, {y0}), {invented:.0%} invented")
    entries = [e for i, e in enumerate(entries)
               if i not in stripe_consumed] + stripe_entries

    # Short-document extension, per row (same y-chaining as
    # group_boxes_into_rows, kept on indices so the flags follow along).
    all_boxes = [e[0] for e in entries]
    order = sorted(range(len(entries)), key=lambda i: all_boxes[i][1])
    tolerance = float(np.mean([b[3] for b in all_boxes])) * 0.5
    rows = [[order[0]]]
    for idx in order[1:]:
        if abs(all_boxes[idx][1] - all_boxes[rows[-1][-1]][1]) < tolerance:
            rows[-1].append(idx)
        else:
            rows.append([idx])
    for row in rows:
        anchors = [i for i in row
                   if all_boxes[i][3] >= GEOMETRY_FULL_RATIO * exp_h]
        if len(anchors) < 2:
            continue
        row_top = int(np.median([all_boxes[i][1] for i in anchors]))
        row_h = int(np.median([all_boxes[i][3] for i in anchors]))
        for i in row:
            x, y, w, h = entries[i][0]
            # Never extend a fragment whose merge was REFUSED - that would
            # quietly repair contested geometry. Merged pages only continue
            # here when their union came out SHORT (sub-band chain).
            if entries[i][2]:
                continue
            if entries[i][1] and not entries[i][3]:
                continue
            short_limit = row_h if entries[i][3] else GEOMETRY_SHORT_RATIO * row_h
            if h >= short_limit:
                continue
            # ...and never widen (steg 10B). The extension is for a page
            # that lost HEIGHT, and its exemption from the invented cap is
            # argued on one page's width - "empty film at worst". A box
            # spanning two pages breaks that argument: card 623_00012's
            # 4110x450 sliver became a 4110x2780 double page at 84 %
            # invented. Such a box keeps its raw geometry so the
            # impossible-geometry guard names it for what it is.
            row_w = int(np.median([all_boxes[j][2] for j in anchors]))
            if w > EXTEND_MAX_WIDTH_RATIO * row_w:
                notes.append(f"NOT extending ({x}, {y}) {w}x{h}: "
                             f"{w / row_w:.1f} pages wide - several pages "
                             "in one box, not a page that lost height")
                continue
            invented = 1.0 - h / row_h
            entries[i][0] = (x, row_top, w, row_h)
            entries[i][1] = True
            notes.append(f"extended short detection at ({x}, {y}) to row "
                         f"height {row_h} ({invented:.0%} invented, "
                         "empty film at worst)")

    return ([e[0] for e in entries], [e[1] for e in entries], notes, refused)


# Page-size prior (architecture addition, Trond 2026-09-08): the page size
# is a KNOWN CONSTANT of the journal format, not a per-blob measurement.
# Calibrated from RAPPORT-2026-09-08-3/-4: healthy detections cluster at
# ~2040-2050 x 2760-2800 full-res pixels across 13 production cards (pitch
# ~2180). If another card format ever appears: measure a healthy card's
# detections the same way (PAGE COORDINATES in any rapport.txt), update the
# prior or run with a per-card estimate - resolve_page_size falls back to
# the estimate LOUDLY whenever no detection lands near the prior, so an
# off-format card never gets silently forced into journal size.
PAGE_SIZE_PRIOR = (2050, PAGE_SIZE_PRIOR_H)
PAGE_SIZE_TOLERANCE = 0.10      # per-card fine-tune bound around the prior
SNAP_PITCH_TOLERANCE = 0.15     # of the pitch: max offset from a grid slot
SNAP_GROWTH_MARK = 0.05         # area growth share that marks a page blue
# Beyond this, a box is not a page at all but several fused into one, and
# shipping it delivers one crop where the journal has four (card 111 in
# background mode: page 1 was 8610x3100 at exit 0, quality 52.5; card 050:
# one 4220x2840 box on an otherwise healthy card). Boxes between the snap
# exemption (1.25) and this stay raw with their loud warning.
SNAP_IMPOSSIBLE_RATIO = 1.5
# How far a row anchor may fall outside the image before it stops being a
# rounding artifact and becomes a misplaced page (steg 9A).
SNAP_EDGE_TOLERANCE = 0.05

# Evidence guard (steg 6A, 2026-09-08, after the full production run of 88
# cards). A card must be FOUND, not composed: card 612130000203_00012
# shipped 41 empty crops built from 9 detections at quality 84.8 GOOD,
# because witnesses and the raster laid out the rest. Counting witnesses,
# snap growth and repairs as "invented" cannot separate that from a healthy
# card - measured, it gives 95 % for 203 but also 85 % for 135 and 94 % for
# 630, which are correct. Snap growth is normal operation (C15); those pages
# exist. What separates is how many pages come out per detection that went
# in: the sick cards run 2.0-4.6, the healthy ones 0.8-1.3.
EVIDENCE_MAX_PAGES_PER_DETECTION = 1.5
# ...and an explicit floor, so a card is never laid out from one or two
# blobs even if the ratio happens to stay under the bound. It only bites
# when something WAS invented: the real journal fasit card has 2 detections
# and 2 pages and invents nothing.
EVIDENCE_MIN_DETECTIONS = 3

# Coverage guard (mandatory, 2026-09-09): field card 612130000036 scored
# 100.0 with its whole first page row OUTSIDE every box. Foreground mass
# outside all page boxes caps the quality score and warns loudly - in every
# mode, because it is the one signal that survives any upstream mistake.
COVERAGE_WARN_SHARE = 0.15      # of total foreground mass

# A position witness must carry real mass: a 20x10 speck of dirt beside the
# pages on the REAL fasit card claimed a phantom page cell before this
# floor existed. Genuine detection rests measure 1.5-2.5% of a page.
WITNESS_MIN_AREA_SHARE = 0.005  # of the page area


def can_hold_two_pages(box, page_w, page_h):
    """Could this detection contain more than one page? Two pages side by
    side span ~2.1 page widths (field pitch 2180 against a 2050 page), so a
    box within SNAP_IMPOSSIBLE_RATIO of a single page holds exactly one -
    and splitting it can only cut that page into pieces. Card 098 split
    sixteen single pages into 2-4 fragments each, because the split pass
    judged them against the MEDIAN of a box list its own fragments
    dominated. The page size is a format constant (C15); use it."""
    return (box[2] > SNAP_IMPOSSIBLE_RATIO * page_w
            or box[3] > SNAP_IMPOSSIBLE_RATIO * page_h)


def suspected_merged_boxes(boxes, page_w, page_h):
    """[(box, how many pages it looks like)] for boxes that still look fused
    after the split pass. Measured against the page size for the same reason
    as above: card 098 reported 38 normal pages as '~2 fused'."""
    return [(b, max(round(b[2] / page_w), round(b[3] / page_h)))
            for b in boxes if can_hold_two_pages(b, page_w, page_h)]


def foreground_outside_boxes(binary_img, boxes):
    """Share of the binary's foreground mass not covered by any box
    (detect-scale boxes). The operator-facing 'did the boxes cover what the
    threshold saw' number."""
    total = int(np.count_nonzero(binary_img))
    if total == 0:
        return 0.0
    h, w = binary_img.shape
    mask = np.zeros((h, w), np.uint8)
    for (x, y, bw, bh) in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    outside = int(np.count_nonzero(binary_img[mask == 0]))
    return outside / total


def resolve_page_size(boxes):
    """Card-level page size: the prior, fine-tuned by the detections that
    already match it (median of those, so strips and fragments do not vote).
    Off-format cards (fixtures, unknown formats) fall back to the per-card
    estimate with a loud note. Size never comes from a single blob."""
    pw0, ph0 = PAGE_SIZE_PRIOR
    good = [b for b in boxes
            if abs(b[2] - pw0) <= PAGE_SIZE_TOLERANCE * pw0
            and abs(b[3] - ph0) <= PAGE_SIZE_TOLERANCE * ph0]
    if len(good) >= 2:
        return (int(np.median([b[2] for b in good])),
                int(np.median([b[3] for b in good])), None)
    if len(good) == 1:
        # One pristine witness: its dimensions ARE this card's page, clamped
        # into the prior band. Using the raw prior instead was measured to
        # SHRINK pages on a card whose true size sits at the band's edge
        # (bottom 200px of content cut) - the sickest cards are exactly
        # where this matters. Announced (steg 4C): one witness out of many
        # detections is a thin basis for a whole card.
        lo_w, hi_w = pw0 * (1 - PAGE_SIZE_TOLERANCE), pw0 * (1 + PAGE_SIZE_TOLERANCE)
        lo_h, hi_h = ph0 * (1 - PAGE_SIZE_TOLERANCE), ph0 * (1 + PAGE_SIZE_TOLERANCE)
        w1 = int(min(max(good[0][2], lo_w), hi_w))
        h1 = int(min(max(good[0][3], lo_h), hi_h))
        return w1, h1, (f"Page size from a SINGLE witness: 1 of "
                        f"{len(boxes)} detections matches the prior "
                        f"{pw0}x{ph0} (+-{PAGE_SIZE_TOLERANCE:.0%}); using "
                        f"{w1}x{h1}")
    ph = int(expected_page_height(boxes))
    # Width fallback: median width of the FULL-HEIGHT boxes. The height
    # estimator can lean on "the tallest box is a whole page" (fragments are
    # shorter), but the widest box may be a fused ROW - wider than a page -
    # so a max-anchored width is wrong in this direction.
    full_h = [b[2] for b in boxes if abs(b[3] - ph) <= 0.2 * ph]
    pw = int(np.median(full_h if full_h else [b[2] for b in boxes]))
    med_w = int(np.median([b[2] for b in boxes]))
    med_h = int(np.median([b[3] for b in boxes]))
    return pw, ph, (
        f"Page-size prior {pw0}x{ph0} not matched by this card - "
        f"0 of {len(boxes)} detections fall within "
        f"{PAGE_SIZE_TOLERANCE:.0%} of it (this card measures a median "
        f"{med_w}x{med_h}, {abs(med_h - ph0) / ph0:.0%} off in height); "
        f"using per-card estimate {pw}x{ph}. A whole card built on a size "
        "the format does not have means the DETECTIONS are short - look "
        "upstream (structure rows cutting pages, washed content), not here")


WITNESS_MIN_DIM_SHARE = 0.05    # of page width AND height: a 50 px sleeve
                                # edge (field card 135) is not a page rest


def snap_pages(boxes, page_w, page_h, flags=None, witnesses=(), stripes=(),
               image_w=None, image_h=None):
    """The final geometry pass: every accepted detection becomes a full page
    box. The blob gives position, the page size gives the dimensions, and
    the row's grid (phase + pitch) decides which page a partial detection
    belongs to - assignment is by CELL (nearest grid slot to the detection
    center), not by gap-chaining, because a right-hand strip of one page
    can sit closer to its neighbour page than to its own sibling strip
    (production card 612130000029, detections 10+11+12).

    Refused = detections that STRADDLE a cell boundary (bridging two pages'
    spans beyond tolerance) - the one geometry no page explains. Their raw
    box is kept in the output so the failing card can be inspected.

    Witnesses are sub-min-size blobs: they never build rows and never vote
    on phase, pitch or row edges - but a witness inside an otherwise EMPTY
    cell of an existing row claims a full page there (field card
    612130000098 lost half a row to the min-size filter). A witness page
    takes its ROW's anchor y (steg 3: the band midpoint put 098's four
    witness pages 1082 px below their row), must be page-like in both
    dimensions, and its cell must lie inside the image (image_w) and inside
    the card's observed column raster (135: a 50x1990 sleeve edge claimed a
    13th column reaching past the image edge).

    Stripes (steg 2, 2026-09-08, Trond's architecture) are the dark
    edge-to-edge bands between page rows that remove_structure_rows deleted,
    as full-res y-intervals. A row's pages must lie in the SLOT between the
    stripe above and the stripe below - so an anchor-less row is anchored
    on whichever edge keeps its box inside the slot (field card
    612130000111: only the TOPS of row 2 survived, and unconditional
    bottom-anchoring stacked the row on row 1). Two invariants are enforced
    on the result and refuse the card when broken: no two page boxes
    overlap, and no page box crosses a stripe.

    Returns (snapped_boxes, flags, notes, refused).
    """
    n = len(boxes)
    flags = list(flags) if flags is not None else [False] * n
    if n == 0:
        return [], [], [], []
    stripes = sorted((int(a), int(b)) for a, b in stripes)

    def inside(y):
        """The image edge is a clamp like any other (steg 9A): card
        612130000623_00024 shipped three pages at y = -400 because the slot
        clamp had no image edge to clamp against."""
        if image_h is None:
            return y
        return int(min(max(y, 0), max(0, image_h - page_h)))

    def slot_around(y_center):
        """(top, bottom) of the open band between the stripes surrounding
        y_center; None on a side with no stripe there."""
        top = max((b for a, b in stripes if b <= y_center), default=None)
        bottom = min((a for a, b in stripes if a >= y_center), default=None)
        return top, bottom

    def fits(y, slot):
        top, bottom = slot
        return ((top is None or y >= top)
                and (bottom is None or y + page_h <= bottom))

    # Rows by transitive y-OVERLAP clustering (2026-09-09, after field card
    # 612130000036): same-row members overlap each other substantially -
    # fragments overlap their full-height anchors - while different rows do
    # not overlap at all. Top-banding cannot do this: a washed row's
    # detections have LOW tops, so the next row fell inside its page-height
    # span and every cell got y-anchored one row down (036); and bottom
    # fragments formed their own band, stacking a phantom row of full pages
    # on the real one (111).
    def y_overlap(a, b):
        lo = max(a[1], b[1])
        hi = min(a[1] + a[3], b[1] + b[3])
        return hi - lo

    # Exempt BEFORE clustering (review finding 2026-09-09): a double-height
    # unsplittable merger overlaps both neighbouring rows and would glue
    # them into one band transitively - the 036 collapse through the back
    # door. Exempt boxes take no part in rows, cells or consensus; they
    # pass through raw further down.
    exempt = {i for i in range(n)
              if boxes[i][2] > 1.25 * page_w or boxes[i][3] > 1.25 * page_h}

    order = sorted((i for i in range(n) if i not in exempt),
                   key=lambda i: boxes[i][1])
    rows = []
    for i in order:
        placed = False
        for row in rows:
            if any(y_overlap(boxes[i], boxes[j])
                   >= 0.4 * min(boxes[i][3], boxes[j][3]) for j in row):
                row.append(i)
                placed = True
                break
        if not placed:
            rows.append([i])
    rows.sort(key=lambda row: min(boxes[i][1] for i in row))

    # Pitch is a property of the physical jacket, shared by all rows (rows
    # start where they start, but the frame raster is one grid).
    # A neighbour distance smaller than the page width is physically
    # impossible as a pitch (pages would overlap) - offset fragments
    # interleaved with their pages produce exactly such false diffs.
    diffs = []
    for row in rows:
        xs = sorted(boxes[i][0] for i in row)
        diffs += [b - a for a, b in zip(xs, xs[1:])
                  if page_w * 1.0 <= b - a <= page_w * 1.45]
    pitch = float(np.median(diffs)) if diffs else page_w * 1.05
    tol = SNAP_PITCH_TOLERANCE * pitch

    snapped, out_flags, notes, refused = [], [], [], []
    origins = []   # detection indices behind each snapped PAGE (not exempt)
    src_idx = []   # detection indices behind EVERY entry, for reporting
    row_ctx = []   # per-row grid/anchor, for the witness pass
    for i in sorted(exempt):
        snapped.append(boxes[i])
        out_flags.append(bool(flags[i]))
        origins.append(None)
        src_idx.append((i,))

    for row in rows:
        members = [i for i in row if i not in exempt]
        if not members:
            continue
        row_boxes = [boxes[i] for i in members]

        fulls = [b for b in row_boxes if b[3] >= 0.85 * page_h]
        anchors = fulls or row_boxes
        row_top = float(np.median([b[1] for b in anchors]))
        row_bottom = float(np.median([b[1] + b[3] for b in anchors]))
        y_lo = min(row_top, row_bottom - page_h)
        y_hi = max(row_top, row_bottom - page_h)
        slot = slot_around(float(np.median([b[1] + b[3] / 2
                                            for b in row_boxes])))
        slot_h = ((slot[1] - slot[0]) if None not in slot else None)

        # Grid phase from the members whose width already matches a page -
        # strips must not vote, their x0 is not a page edge. A row with no
        # such witness anchors on the strips' left edges instead (logged
        # implicitly by their growth notes).
        trusted = [b for b in row_boxes
                   if abs(b[2] - page_w) <= 0.08 * page_w]
        phase_src = trusted or row_boxes
        ref = phase_src[0][0]
        offsets = []
        for b in phase_src:
            d = (b[0] - ref) % pitch
            if d > pitch / 2:
                d -= pitch
            offsets.append(d)
        phase = ref + float(np.median(offsets))

        # Cell assignment by detection center. A member may reach into the
        # INTER-PAGE GAP (a dirty seam puts the split cut mid-gap rather
        # than on the page edge) but never into the neighbour PAGE - that
        # is the straddle no single page explains.
        gap = max(0.0, pitch - page_w)
        cells = {}
        for i in members:
            x0, y0, w, h = boxes[i]
            k = round((x0 + w / 2 - phase - page_w / 2) / pitch)
            cell_x = phase + k * pitch
            if (x0 < cell_x - gap - tol
                    or x0 + w > cell_x + page_w + gap + tol):
                refused.append((i,))
                notes.append(f"REFUSED snap of detection {i + 1}: spans "
                             "beyond one grid cell (bridges two pages)")
                snapped.append(boxes[i])
                out_flags.append(bool(flags[i]))
                origins.append(None)
                src_idx.append((i,))
                continue
            cells.setdefault(k, []).append(i)

        for k in sorted(cells):
            group = cells[k]
            x = int(round(phase + k * pitch))
            g_top = min(boxes[i][1] for i in group)
            g_bottom = max(boxes[i][1] + boxes[i][3] for i in group)
            # The credible edge: whichever of the group's top/bottom agrees
            # better with the row consensus (washed-out tops leave the
            # BOTTOM as the surviving edge; tilted cards need the group's
            # own edge rather than one row-wide y).
            top_err = abs(g_top - row_top)
            bottom_err = abs(g_bottom - row_bottom)
            # An anchor-less row (no full-height member) has lost one edge
            # - which one, the STRIPES decide: the box must sit inside its
            # slot. Bottoms first (036: washed tops, bottoms survived),
            # tops when only that fits (111: washed bottoms, tops survived).
            # Without stripe evidence the bottom rule stands, and the
            # overlap invariant below catches the wrong guess.
            if not fulls:
                candidates = [g_bottom - page_h, g_top]
                y = next((c for c in candidates if fits(c, slot)),
                         candidates[0])
            else:
                y = g_top if top_err <= bottom_err else g_bottom - page_h
            y_before_edge = int(min(max(y, y_lo), y_hi))
            y = inside(y_before_edge)
            if abs(y - y_before_edge) > SNAP_EDGE_TOLERANCE * page_h:
                dets = "+".join(str(i + 1) for i in sorted(group))
                refused.append(tuple(sorted(group)))
                notes.append(f"REFUSED: detections {dets} anchor to "
                             f"y={y_before_edge}, outside the image "
                             f"(0-{image_h}) - moved to {y} to keep the box "
                             "on the card, but that is not where the "
                             "evidence put it")
            if slot_h is not None and slot_h < page_h:
                dets = "+".join(str(i + 1) for i in sorted(group))
                refused.append(tuple(sorted(group)))
                notes.append(f"REFUSED snap of detections {dets}: the slot "
                             f"between stripes {slot[0]}-{slot[1]} is "
                             f"{slot_h} px, shorter than a page ({page_h})")
                for i in sorted(group):
                    snapped.append(boxes[i])
                    out_flags.append(bool(flags[i]))
                    origins.append(None)
                    src_idx.append((i,))
                continue
            if not fits(y, slot):
                # A full box that would cross a stripe: keep it in the slot.
                if slot[0] is not None:
                    y = max(y, slot[0])
                if slot[1] is not None:
                    y = min(y, slot[1] - page_h)
                y = inside(y)
            covered = sum(boxes[i][2] * boxes[i][3] for i in group)
            grown = 1.0 - min(1.0, covered / (page_w * page_h))
            is_grown = grown > SNAP_GROWTH_MARK
            snapped.append((x, y, page_w, page_h))
            out_flags.append(bool(any(flags[i] for i in group) or is_grown))
            origins.append(tuple(sorted(group)))
            src_idx.append(tuple(sorted(group)))
            if is_grown:
                dets = "+".join(str(i + 1) for i in sorted(group))
                notes.append(f"snapped detections {dets} to full page at "
                             f"({x}, {y}) - {grown:.0%} of the page area "
                             "grown to the known size")

        # Row context for the witness pass below: the row's y-envelope, its
        # grid, and the anchor its cells actually got.
        row_y0 = min(boxes[i][1] for i in members)
        row_y1 = max(boxes[i][1] + boxes[i][3] for i in members)
        row_ys = [snapped[k][1] for k, o in enumerate(origins)
                  if o and set(o) <= set(members)]
        row_ctx.append(dict(y0=row_y0, y1=row_y1, phase=phase, cells=set(cells),
                            anchor=(int(np.median(row_ys)) if row_ys
                                    else int(round((y_lo + y_hi) / 2))),
                            slot=slot))

    # Position witnesses: a sub-min blob whose center falls inside a row's
    # y-envelope and inside an otherwise EMPTY cell proves the cell holds a
    # page. Runs after every row is placed so the card's column raster is
    # known: a cell must lie inside the image and inside [min x, max x] of
    # the pages the card actually has.
    # The raster is cross-row evidence: with a single row there is nothing
    # to compare against (098's own case is half a row of rests LEFT of
    # every detected page), so it only binds on cards with >= 2 rows.
    page_xs = [snapped[k][0] for k, o in enumerate(origins) if o]
    raster = ((min(page_xs), max(page_xs))
              if page_xs and len(row_ctx) >= 2 else None)
    for (wx, wy, ww, wh) in witnesses:
        if ww * wh < WITNESS_MIN_AREA_SHARE * page_w * page_h:
            continue
        cy = wy + wh / 2
        ctx = next((r for r in row_ctx if r["y0"] <= cy <= r["y1"]), None)
        if ctx is None:
            continue
        k = round((wx + ww / 2 - ctx["phase"] - page_w / 2) / pitch)
        if k in ctx["cells"]:
            continue
        x = int(round(ctx["phase"] + k * pitch))
        why = None
        if ww < WITNESS_MIN_DIM_SHARE * page_w or wh < WITNESS_MIN_DIM_SHARE * page_h:
            why = (f"a {ww}x{wh} rest is not page-like (needs >= "
                   f"{WITNESS_MIN_DIM_SHARE:.0%} of the page in both directions)")
        elif x < 0 or (image_w is not None and x + page_w > image_w):
            why = f"cell {x}-{x + page_w} reaches outside the image"
        elif raster and not (raster[0] - tol <= x <= raster[1] + tol):
            why = (f"cell at x={x} lies outside the card's column raster "
                   f"{raster[0]}-{raster[1]}")
        if why:
            notes.append(f"position witness at ({wx}, {wy}) ignored: {why}")
            continue
        ctx["cells"].add(k)
        y = ctx["anchor"]
        if not fits(y, ctx["slot"]):
            if ctx["slot"][0] is not None:
                y = max(y, ctx["slot"][0])
            if ctx["slot"][1] is not None:
                y = min(y, ctx["slot"][1] - page_h)
        snapped.append((x, inside(y), page_w, page_h))
        out_flags.append(True)
        origins.append(())
        src_idx.append(())
        notes.append(f"page from position witness at ({x}, {y}) - a "
                     f"{ww}x{wh} rest proves the cell holds a page")

    # Impossible geometry (steg 4B): several pages fused into one box. The
    # snap exempts anything over 1.25 pages and passes it through raw, so
    # without this a four-page box shipped at exit 0 (card 111 background).
    for k, (x, y, w, h) in enumerate(snapped):
        if w > SNAP_IMPOSSIBLE_RATIO * page_w or h > SNAP_IMPOSSIBLE_RATIO * page_h:
            refused.append(tuple(src_idx[k]))
            notes.append(f"REFUSED: box at ({x}, {y}) is {w}x{h} - "
                         f"impossible geometry, over "
                         f"{SNAP_IMPOSSIBLE_RATIO:g} pages "
                         f"({page_w}x{page_h}); several pages in one box")

    # Invariants (Trond, 2026-09-08): no two page boxes overlap, no page box
    # crosses a stripe. A violation is a wrong guess somewhere above, and
    # it must fail the card - never ship as a quiet page list (card 111
    # scored align 100 / quality 90 with a whole row stacked on another).
    pages = [k for k, o in enumerate(origins) if o is not None]
    for a_i, a in enumerate(pages):
        ax, ay, aw, ah = snapped[a]
        for b in pages[a_i + 1:]:
            bx, by, bw, bh = snapped[b]
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox > 0 and oy > 0:
                refused.append(tuple(sorted(set(origins[a] + origins[b]))))
                notes.append(f"REFUSED: page boxes at ({ax}, {ay}) and "
                             f"({bx}, {by}) overlap by {ox}x{oy} px - "
                             "impossible geometry")
    for k in pages:
        x, y, w, h = snapped[k]
        for a, b in stripes:
            if y < b and y + h > a:
                refused.append(tuple(origins[k]))
                notes.append(f"REFUSED: page box at ({x}, {y}) crosses the "
                             f"stripe {a}-{b}")
                break
    return snapped, out_flags, notes, refused


class ChainResult(NamedTuple):
    """What the coordinate-only chain decided, and what main must print."""
    boxes: list
    repaired: set          # indices of pages a repair or the snap touched
    fragment_groups: list
    refused_groups: list
    snap_refused: list
    geo_overload: bool
    substantial: int       # substantial repairs, for the overload message
    quality: dict          # of the FINAL boxes; None when nothing changed
    output: list           # (stream, text) in the order they were printed
    card_refusals: list    # reasons this card must not ship (steg 6A/6B)
    layout_refused: bool   # ...and whether a LAYOUT invariant was broken:
                           # impossible layout is its own evidence that the
                           # threshold is wrong (steg 9E), and it can happen
                           # at a LOW border share when detection splinters
                           # instead of vanishing (card 647: border 26 %).
    evidence_refused: bool # ...and whether the EVIDENCE guard was one of
                           # them. An explicit flag, not a search for words
                           # in a human sentence: the staircase's trigger
                           # depends on it, and re-wording a layout message
                           # must never be able to change control flow.


def repair_and_snap(boxes, witnesses=(), stripes=(), image_w=None,
                    image_h=None, header_px=0, page_prior=PAGE_SIZE_PRIOR):
    """The coordinate-only part of the chain: geometric completion, the
    over-repair judgement, the snap to the format page size, and the
    fragment guard's re-check.

    Nothing here reads the image, so a report carrying the PRE-REPAIR
    detections (steg 5C) can be replayed through exactly this code and
    reproduce what production shipped. main() prints the result and takes
    the exit decision; this function decides the geometry. Extracted
    verbatim from main 2026-09-08 - the printing order is preserved by
    collecting the lines rather than emitting them here.
    """
    out = []
    refusals = []
    evidence_refusals = []
    boxes_in = list(boxes)

    # Header content is not a page row (steg 9B). Dropped BEFORE anything
    # counts detections, so the evidence guard judges pages against pages.
    if header_px:
        hdr_w, hdr_h, _ = resolve_page_size(boxes)
        header_boxes, header_note = header_zone_detections(
            boxes, hdr_w, hdr_h, header_px)
        if header_note:
            out.append((1, f"\n{header_note}"))
        for b in header_boxes:
            out.append((1, f"  dropped header content at ({b[0]}, {b[1]}) "
                           f"size {b[2]} x {b[3]}"))
        if header_boxes:
            boxes = [b for b in boxes if b not in header_boxes]

    detections_in = len(boxes)

    boxes, geo_flags, geo_notes, refused_groups = complete_geometry(boxes)
    repaired_count = sum(geo_flags)
    if geo_notes:
        out.append((1, f"\n{repaired_count} pages geometry-completed:"))
        for note in geo_notes:
            out.append((1, f"  {note}"))
        # Repair changed boxes and counts: re-sort with the flags riding
        # along (row grouping only reads elements 0/1/3, so the flag can sit
        # at index 4), then rescore - the score must describe what ships.
        tagged = sort_boxes_by_rows(
            [b + (f,) for b, f in zip(boxes, geo_flags)])
        boxes = [t[:4] for t in tagged]
        geo_flags = [t[4] for t in tagged]
    geo_indices = {i for i, f in enumerate(geo_flags) if f}

    # Card-level sanity: when geometry has to save more than half the card,
    # the card is genuinely sick - repair must not become silent success.
    # Only SUBSTANTIAL repairs count toward condemning a card: a merge
    # that tiles its union perfectly (0% invented - split-pass churn, clean
    # fragment pairs) is bookkeeping, not fabrication. And at least 3 of
    # them - on a one- or two-page card any single repair is already "most
    # of the card".
    substantial = sum(1 for m in re.findall(r"(\d+)% invented",
                                            "\n".join(geo_notes))
                      if int(m) >= 3)
    geo_overload = (substantial >= 3 and
                    substantial > GEOMETRY_MAX_REPAIR_SHARE * len(boxes))
    if geo_overload:
        out.append((2, f"\nERROR: geometry had to repair {substantial} of "
                       f"{len(boxes)} pages substantially "
                       f"(limit {GEOMETRY_MAX_REPAIR_SHARE:.0%}) - card is "
                       "sick, not repairable"))

    # Snap to the known page size (architecture addition, 2026-09-08 - the
    # format's page size is a constant; blobs give position, the prior
    # gives size). Runs BEFORE the guard's re-check: cell assignment
    # reunites fragments the chain criteria cannot (a wash wider than the
    # gap allowance splits a page into pieces the chains refuse to link,
    # and the extension pass alone would leave the sibling as a ghost).
    # Skipped on an already-failing card - it keeps raw geometry for
    # diagnosis. Snap growth does NOT count toward the over-repair limit:
    # normalizing to the known size is normal operation, and the coming
    # occupancy check is the content verification, not this threshold.
    snap_refused = []
    if not (refused_groups or geo_overload):
        geo_flags_list = [i in geo_indices for i in range(len(boxes))]
        page_w, page_h, size_note = resolve_page_size(boxes)
        if size_note:
            out.append((1, f"\n{size_note}"))
        boxes, snap_flags, snap_notes, snap_refused = snap_pages(
            boxes, page_w, page_h, flags=geo_flags_list,
            witnesses=witnesses, stripes=stripes, image_w=image_w,
            image_h=image_h)
        snapped_count = sum(1 for note in snap_notes
                            if note.startswith("snapped"))
        if snap_notes:
            out.append((1, f"\n{snapped_count} pages snapped to page size "
                           f"{page_w} x {page_h}:"))
            for note in snap_notes:
                out.append((1, f"  {note}"))
        tagged = sort_boxes_by_rows(
            [b + (fl,) for b, fl in zip(boxes, snap_flags)])
        boxes = [t[:4] for t in tagged]
        geo_indices = {i for i, t in enumerate(tagged) if t[4]}
        if snap_refused:
            out.append((2, f"\nERROR: {len(snap_refused)} suspected page "
                           "fragment group(s) - detections irreconcilable "
                           "with the page grid:"))
            for group in snap_refused:
                out.append((2, "  detections "
                               + "+".join(str(i + 1) for i in group)))

    # Fragment guard re-check, now on the SNAPPED geometry: uniform pages
    # normally leave it nothing, so what it finds is a genuine leftover
    # (e.g. stacks involving the snap-exempt merged boxes). Refused merges
    # fail the card via refused_groups regardless.
    fragment_groups = find_fragment_groups(boxes)
    if fragment_groups:
        out.append((2, f"\nERROR: {len(fragment_groups)} suspected page "
                       "fragment group(s) - stacked detections that do not "
                       "reconcile with the page grid:"))
        for group in fragment_groups:
            pages = "+".join(str(i + 1) for i in group)
            coords = ", ".join(str(boxes[i]) for i in group)
            out.append((2, f"  pages {pages}: {coords}"))

    # Evidence guard (steg 6A): a card must be found, not composed.
    ratio = len(boxes) / max(1, detections_in)
    out.append((1, f"\nEvidence: {len(boxes)} pages from {detections_in} "
                   f"detections (ratio {ratio:.2f}, refused above "
                   f"{EVIDENCE_MAX_PAGES_PER_DETECTION:g})"))
    if ratio > EVIDENCE_MAX_PAGES_PER_DETECTION:
        evidence_refusals.append(
            f"{len(boxes)} pages laid out from only {detections_in} "
            f"detections (ratio {ratio:.2f}, limit "
            f"{EVIDENCE_MAX_PAGES_PER_DETECTION:g}) - the card was composed "
            "from rests, not read")
    if detections_in < EVIDENCE_MIN_DETECTIONS and len(boxes) > detections_in:
        evidence_refusals.append(
            f"too little evidence to lay out a card: {detections_in} "
            f"detection(s) (minimum {EVIDENCE_MIN_DETECTIONS}) AND "
            f"{len(boxes) - detections_in} page(s) invented on top of them")

    refusals.extend(evidence_refusals)
    layout_refusals = []

    # Layout invariants (steg 6B): a physical card holds at most MAX_ROWS
    # rows of at most MAX_PAGES_PER_ROW pages. These were warnings until
    # 612130000432_00024 shipped 60 pages in SIX rows at quality 71.1.
    layout_rows = group_boxes_into_rows(boxes)
    if len(layout_rows) > MAX_ROWS:
        layout_refusals.append(f"{len(layout_rows)} rows detected - a card "
                               f"holds at most {MAX_ROWS}")
    for i, row in enumerate(layout_rows, 1):
        if len(row) > MAX_PAGES_PER_ROW:
            layout_refusals.append(f"{len(row)} pages in one row (row {i}) - "
                                   f"a card holds at most "
                                   f"{MAX_PAGES_PER_ROW}")
    refusals.extend(layout_refusals)
    for reason in refusals:
        out.append((2, f"\nERROR: {reason}"))

    # Quality is computed ONCE, here, on the boxes that actually ship (steg
    # 10A). Recomputing it at each step invited staleness: 9A compared
    # against a name the snap had already rebound and silently kept the
    # pre-snap score, sending two healthy controls to SVAK. One place, one
    # answer, and the caller's contour-based score stands when nothing
    # changed.
    quality = compute_card_quality(boxes, None) if boxes != boxes_in else None

    return ChainResult(boxes, geo_indices, fragment_groups, refused_groups,
                       snap_refused, geo_overload, substantial, quality, out,
                       refusals, bool(layout_refusals),
                       bool(evidence_refusals))


# How far below the header mask a detection may start and still be the
# header's own content (steg 9B). Measured on all four affected cards: the
# header blobs start 0-250 px below the mask line, while the first real page
# row starts 1690-2670 px below it. A quarter of a page height sits in that
# gap with room on both sides.
HEADER_ZONE_REACH = 0.25


def header_zone_detections(boxes, page_w, page_h, header_px):
    """([detections that are header content, not pages], note).

    After step two the top band is often gone from the structure list -
    three of the four affected field cards had only the bottom band left -
    and the header text below the mask line then stands as detections and
    grows the card a sixth row. We know where the header is, so we use it.

    A detection is header content only if all of these hold:
      - it starts within HEADER_ZONE_REACH of the mask line (a header blob
        continues the masked band; a page row starts a row gap below it -
        this is what keeps a SHORT first row from being eaten),
      - its centre lies above the topmost row carrying at least two
        full-height boxes (no such row: no anchor, so nothing is dropped),
      - it matches the page prior in NEITHER dimension. Width alone is
        enough to save it: a first page row that crosses the mask keeps its
        page WIDTH while the mask cuts its top (that fixture exists). So is
        height alone: a fused row is several pages wide but page-high, and
        dropping it would lose four pages silently instead of failing
        loudly. Measured on all eight field blobs, none matches either -
        they run 1.3-4.8 page widths at 0.46-0.81 page heights.
    """
    if not boxes or header_px <= 0:
        return [], None
    rows = group_boxes_into_rows(boxes)
    anchor_top = None
    for row in rows:
        full = [b for b in row if b[3] >= 0.85 * page_h]
        if len(full) >= 2:
            anchor_top = min(b[1] for b in full)
            break
    if anchor_top is None:
        return [], ("header zone: no full-height row to anchor on - nothing "
                    "dropped")
    reach = header_px + HEADER_ZONE_REACH * page_h
    dropped = []
    for b in boxes:
        x, y, w, h = b
        if y >= reach:
            continue
        if y + h / 2 >= anchor_top:
            continue
        if (abs(h - page_h) <= 0.10 * page_h
                or abs(w - page_w) <= 0.10 * page_w):
            continue          # page-shaped in either dimension: not header
        dropped.append(b)
    note = (f"header zone: first page row at y={anchor_top}, dropping "
            f"{len(dropped)} detection(s) starting above "
            f"{int(reach)} (mask {header_px})" if dropped else
            f"header zone: first page row at y={anchor_top}, nothing above it")
    return dropped, note


def read_manual_boxes(path):
    """[(x, y, w, h)] from the operator's CSV, in the order given.

    One line per page, full-resolution integers, no header line (steg 11).
    The order IS the page numbering, so nothing here sorts or normalises -
    the whole point of manual mode is that what the operator drew is what
    gets cut.
    """
    boxes = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"{path} line {n}: expected 4 numbers "
                             f"(x,y,w,h), got {len(parts)}: {line!r}")
        try:
            boxes.append(tuple(int(p) for p in parts))
        except ValueError:
            raise ValueError(f"{path} line {n}: not whole numbers: {line!r}")
    return boxes


def validate_manual_boxes(boxes, image_w, image_h, warn=False):
    """Problems that must stop the run (warn=False), or warnings that must
    not (warn=True). A box outside the image or without area cannot be cut;
    overlapping boxes can, and the operator may well have meant them."""
    if warn:
        out = []
        for i, a in enumerate(boxes, 1):
            for j, b in enumerate(boxes[i:], i + 1):
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ox > 0 and oy > 0:
                    out.append(f"pages {i} and {j} overlap by {ox}x{oy} px")
        return out
    problems = []
    for i, (x, y, w, h) in enumerate(boxes, 1):
        if w <= 0:
            problems.append(f"page {i}: width {w} is not positive")
        if h <= 0:
            problems.append(f"page {i}: height {h} is not positive")
        if w > 0 and h > 0 and not (0 <= x and 0 <= y
                                    and x + w <= image_w
                                    and y + h <= image_h):
            problems.append(f"page {i}: box ({x}, {y}, {w}, {h}) reaches "
                            f"outside the image (0-{image_w} x 0-{image_h})")
    return problems


def card_cells(boxes, page_w, page_h, image_w, margin_cells=0,
               omitted_out=None):
    """Every cell of the card's raster: the ones a page occupies AND the
    empty ones beside them (steg 8A).

    The empty cells are the point. Measured over the field reports, the SIZE
    of a witness blob cannot tell a real cell from an empty one (correct
    cards 0.8-9.3 % of a page, card 203 - which shipped empty crops -
    0.5-8.5 %), so the evidence has to be measured per cell instead, and
    that needs known-empty cells as the control. A short row on a card with
    room to spare supplies them by the dozen.

    Phase is per ROW: real cards are rows-only and rows are not vertically
    aligned (Trond, 2026-09-04). Pitch is card-wide - one physical raster.

    Only the raster the ROW ITSELF spans is enumerated (steg 9F). Running
    on to the image edge put the control cells in the card MARGIN: healthy
    card 612130000531_00012 reported its five page=0 cells at x = 50-255,
    left of its first page at 2200, where jacket and frame measure 0.39-0.48
    foreground and mean nothing. margin_cells opts extra cells in on each
    side; omitted_out receives how many were left out.
    """
    if not boxes:
        return []
    rows = group_boxes_into_rows(boxes)
    diffs = []
    for row in rows:
        xs = sorted(b[0] for b in row)
        diffs += [b - a for a, b in zip(xs, xs[1:])
                  if page_w <= b - a <= page_w * 1.45]
    pitch = float(np.median(diffs)) if diffs else page_w * 1.05

    # The raster every row is measured against is the CARD's, not its own
    # (steg 10D): a short bottom row is where the known-empty control cells
    # live, and they only exist if the row is enumerated across the width
    # the other rows span. Card 098's row 5 holds 6 pages of 12, and those
    # six empty cells are real card area with no page in it.
    card_x0 = min(b[0] for row in rows for b in row)
    card_x1 = max(b[0] for row in rows for b in row)
    cells = []
    omitted = 0
    for row in rows:
        xs = sorted(b[0] for b in row)
        y = int(np.median([b[1] for b in row]))
        taken = set(xs)
        k_lo = int(round((card_x0 - xs[0]) / pitch))
        k_hi = int(round((card_x1 - xs[0]) / pitch))
        for k in range(k_lo - margin_cells, k_hi + margin_cells + 1):
            x = int(round(xs[0] + k * pitch))
            if x < 0 or x + page_w > image_w:
                omitted += 1
                continue
            on_page = any(abs(x - px) <= pitch * 0.25 for px in taken)
            cells.append({"row": rows.index(row) + 1, "x": x, "y": y,
                          "page": 1 if on_page else 0})
        # Cells outside the card's own raster are margin, not evidence.
        omitted += max(0, int((image_w - page_w - xs[0]) // pitch) - k_hi)
        omitted += max(0, int(xs[0] // pitch) + k_lo)
    if omitted_out is not None:
        omitted_out.append(omitted)
    return cells


def cell_evidence(gray_small, cell, page_w, page_h, scale, threshold,
                  dark_pages=True):
    """(foreground share, edge share) inside one cell, measured on the
    ORIGINAL graytone rather than on the binary - the binary is exactly what
    failed on the faded cards, so it cannot be its own witness.

    dark_pages follows the run's polarity decision. Without it the measure
    reads 0.000 foreground on every real page of a Yamaha-type card (bright
    pages on a dark card) - which would quietly poison the calibration 8B is
    supposed to draw from these numbers."""
    x0 = int(cell["x"] * scale)
    y0 = int(cell["y"] * scale)
    x1 = min(gray_small.shape[1], int((cell["x"] + page_w) * scale))
    y1 = min(gray_small.shape[0], int((cell["y"] + page_h) * scale))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    region = gray_small[y0:y1, x0:x1]
    fg = float(np.mean(region <= threshold if dark_pages
                       else region >= threshold))
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = float(np.mean(mag > CELL_EDGE_LEVEL))
    return fg, edge


CELL_EDGE_LEVEL = 40.0   # Sobel magnitude counted as an edge (steg 8A,
                         # measurement only - nothing decides on it yet)


def make_anon_mask(shape, contours, dilate_radius):
    """Solid silhouettes of the detected blobs, strictly 0/255.

    Filling each EXTERNAL contour erases everything inside a blob (text is a
    hole in the blob, holes get painted over), while a gap that reaches the
    blob's edge - a stitching seam splitting a page - is not a hole and stays
    visible. That asymmetry is the whole point: geometry out, content not.
    Never smooth or close this mask; it would seal the seam gaps we ship it
    out to reveal.
    """
    mask = np.zeros(shape, np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    if dilate_radius > 0:
        kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def _stamp_solid(img, color, draw):
    """Run `draw` on a fresh mono layer, then paint every marked pixel in one
    solid color. OpenCV antialiases text regardless of lineType, and the
    anonymized output must never contain midtones - so no drawing call may
    touch the output directly."""
    layer = np.zeros(img.shape[:2], np.uint8)
    draw(layer)
    img[layer > 127] = color


def render_anon_viz(mask, boxes_fullres, fullres_to_mask, label, banner_color,
                    fragment_indices=frozenset(), repaired_indices=frozenset()):
    """Annotate the silhouette mask with page boxes and the quality banner.
    Boxes whose index is in fragment_indices are marked orange (one part of
    an irreconcilable suspected split page); repaired_indices are marked
    blue (geometry-completed pages - merged fragments or extended shorts).

    Everything here must keep the two-level safety property: INTER_NEAREST for
    the resize and hard-thresholded stamps for all overlay drawing, so the
    output holds only black/white plus the overlay palette.
    """
    viz_scale = min(1.0, 2000 / max(mask.shape[1], mask.shape[0]))
    viz = cv2.resize(mask, None, fx=viz_scale, fy=viz_scale,
                     interpolation=cv2.INTER_NEAREST)
    viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    fullres_to_viz = fullres_to_mask * viz_scale
    scaled = [(int(x * fullres_to_viz), int(y * fullres_to_viz),
               int(w * fullres_to_viz), int(h * fullres_to_viz))
              for (x, y, w, h) in boxes_fullres]

    def category(i):
        if i in fragment_indices:
            return "fragment"
        if i in repaired_indices:
            return "repaired"
        return "page"

    def draw_boxes(layer, wanted):
        for i, (sx, sy, sw, sh) in enumerate(scaled):
            if category(i) == wanted:
                cv2.rectangle(layer, (sx, sy), (sx + sw, sy + sh), 255, 2)

    def draw_numbers(layer):
        for i, (sx, sy, _, _) in enumerate(scaled, 1):
            cv2.putText(layer, str(i), (sx + 5, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    _stamp_solid(viz, (0, 255, 0), lambda layer: draw_boxes(layer, "page"))
    _stamp_solid(viz, FRAGMENT_MARK_COLOR,
                 lambda layer: draw_boxes(layer, "fragment"))
    _stamp_solid(viz, GEOMETRY_MARK_COLOR,
                 lambda layer: draw_boxes(layer, "repaired"))
    _stamp_solid(viz, (0, 0, 255), draw_numbers)

    banner_h = 32
    banner = np.zeros((banner_h, viz.shape[1], 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    _stamp_solid(banner, banner_color,
                 lambda layer: cv2.putText(layer, label, (8, 22),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                           255, 2))
    return np.vstack([banner, viz])


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


def group_boxes_into_rows(boxes, tolerance_ratio=0.5):
    """Group boxes into rows by y proximity; each row sorted left-to-right.

    Rows are the card's only real axis - there is no column structure, and
    rows start where they start (Trond, 2026-09-04).
    """
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    avg_height = np.mean([b[3] for b in boxes])
    tolerance = avg_height * tolerance_ratio

    rows = [[boxes_sorted[0]]]
    for box in boxes_sorted[1:]:
        if abs(box[1] - rows[-1][-1][1]) < tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [sorted(row, key=lambda b: b[0]) for row in rows]


def sort_boxes_by_rows(boxes, tolerance_ratio=0.5):
    """Sort boxes: top-to-bottom by row, left-to-right within each row."""
    return [box for row in group_boxes_into_rows(boxes, tolerance_ratio)
            for box in row]


def compute_card_quality(boxes, contours):
    """Compute card-level quality score (0-100) for segmentation results.

    Rows are the card's only real axis - there is no column structure, and
    rows are NOT vertically aligned with each other (Trond, 2026-09-04), so
    nothing here may reward or punish column geometry.

    Components:
    - Size consistency (30%): How uniform page sizes are
    - Row alignment (40%): How level the pages sit within each row
    - Spacing regularity (20%): Rhythm of row centers and of in-row gaps
    - Shape regularity (10%): How rectangular the detections are
    """
    if len(boxes) < 2:
        # 'grid' included even here: callers print it unconditionally.
        grid = "1 row: 1" if boxes else "0 rows"
        return {'total': 100.0, 'size': 100.0, 'alignment': 100.0,
                'spacing': 100.0, 'shape': 100.0, 'grid': grid}

    widths = np.array([b[2] for b in boxes])
    heights = np.array([b[3] for b in boxes])

    # --- 1. Size consistency (30%) ---
    w_median = np.median(widths)
    h_median = np.median(heights)
    w_cv = np.std(widths) / w_median if w_median > 0 else 0
    h_cv = np.std(heights) / h_median if h_median > 0 else 0
    size_score = max(0.0, 100.0 * (1.0 - (w_cv + h_cv) * 2.0))

    rows = group_boxes_into_rows(boxes)
    grid = (f"{len(rows)} row" + ("s" if len(rows) != 1 else "") + ": "
            + "+".join(str(len(r)) for r in rows))

    # --- 2. Row alignment (40%): pages in a row sit level ---
    avg_height = np.mean(heights)
    row_spreads = [np.std([b[1] for b in row]) / avg_height
                   for row in rows if len(row) > 1]
    avg_row_spread = np.mean(row_spreads) if row_spreads else 0
    alignment_score = max(0.0, 100.0 * (1.0 - avg_row_spread * 10.0))

    # --- 3. Spacing regularity (20%): row rhythm + in-row gap rhythm ---
    gap_scores = []
    row_centers = sorted(np.mean([b[1] for b in row]) for row in rows)
    row_gaps = np.diff(row_centers)
    if len(row_gaps) > 1 and np.mean(row_gaps) > 0:
        row_gap_cv = np.std(row_gaps) / np.mean(row_gaps)
        gap_scores.append(max(0.0, 100.0 * (1.0 - row_gap_cv * 3.0)))
    else:
        gap_scores.append(100.0)

    in_row_cvs = []
    for row in rows:
        if len(row) >= 3:
            centers = [b[0] + b[2] / 2 for b in row]
            gaps = np.diff(centers)
            if np.mean(gaps) > 0:
                in_row_cvs.append(np.std(gaps) / np.mean(gaps))
    if in_row_cvs:
        gap_scores.append(max(0.0, 100.0 * (1.0 - np.mean(in_row_cvs) * 3.0)))
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
        'grid': grid,
    }


def refine_box_local(input_file, box, otsu_thresh, orig_w, orig_h,
                     invert=False, header_skip_px=0, deviation=None):
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

    # Threshold (same rule as the global pass)
    if deviation is not None:
        level, band = deviation
        binary = ((np.abs(region_np.astype(np.float32) - level) > band)
                  .astype(np.uint8) * 255)
    else:
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



def code_version(repo=None):
    """Short git SHA of the code that is running, read straight from .git
    (no subprocess: Finder launches with no PATH). 'ukjent' when the repo
    metadata is missing, so a report never claims a version it cannot
    prove. Two same-day report sets once differed with nothing in either
    saying which code ran."""
    repo = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            ref_file = repo / ".git" / ref
            if ref_file.exists():
                head = ref_file.read_text().strip()
            else:
                packed = (repo / ".git" / "packed-refs").read_text()
                head = next(line.split()[0] for line in packed.splitlines()
                            if line.endswith(" " + ref))
        return head[:7] if re.fullmatch(r"[0-9a-f]{40,64}", head) else "ukjent"
    except (OSError, StopIteration, IndexError):
        return "ukjent"


def run_mode(args):
    """The word for the binarization mode, as printed in every report."""
    return "bakgrunn-foerst" if args.background_first else "standard"


def finish_card(boxes_fullres, input_file, input_path, out_dir,
                original_width, original_height, args, padding=None,
                header=True, start_number=1):
    """Cut the pages, write the header band, drop the sentinel,
    archive the panorama.

    The tail of a run, shared by the automatic path and manual mode
    (steg 11) so the two can never drift apart on what matters
    downstream: page numbering, the `_done` ordering (C2), the header
    page (C11) and archiving (C13). padding overrides args.padding -
    manual mode passes 0, because the operator drew what he wanted
    cut. header=False and start_number=0 belong together: in manual mode
    the operator drew the header too, as the FIRST box, so it is cut like
    any other page and lands as page_000 by numbering rather than by a
    separate downscaled pass.
    """
    pages_dir = out_dir / "pages"

    # === STEP 6: Extract pages (optional) ===
    if not args.skip_extraction:
        print("\n=== EXTRACTING PAGES (parallel) ===")
        pages_dir.mkdir(exist_ok=True)

        # Number of parallel workers
        num_workers = 5
        print(f"Using {num_workers} parallel workers")

        # Crop margin: a fraction of the median page unless given in pixels
        if padding is not None:
            pad_x = pad_y = int(padding)
        elif args.padding is None:
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
            for i, (x, y, w, h) in enumerate(boxes_fullres, start_number)
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

        # Keep the header band as page zero. Everything above the first page
        # row IS the header - on the real journal cards the typed text sits
        # below the fixed mask line (2026-09-04: a ratio-only crop cut the
        # date and card index in half), so the crop extends to the topmost
        # detected page. Capped at twice the configured band so a sparse card
        # cannot swallow empty rows into page_000; never less than the band.
        header_px = int(original_height * args.header_skip)
        if boxes_fullres:
            first_page_y = min(b[1] for b in boxes_fullres)
            header_px = max(header_px, min(first_page_y, header_px * 2))
        if header and not args.no_header_page and header_px > 0:
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


def main(otsu_override=None, step2=False, step1_border=None,
         step1_reasons=()):
    parser = argparse.ArgumentParser(description='Segment microfiche pages')
    # Not argparse-required: a missing input must exit 1 (generic failure),
    # while argparse errors exit 2 and would collide with EXIT_NO_PAGES.
    parser.add_argument('--input', '-i', default=None, help='Input image file')
    parser.add_argument('--order', '-o', choices=['columns', 'rows'], default=READING_ORDER,
                        help='Reading order: columns (down then right) or rows (right then down)')
    parser.add_argument('--header-skip', '-hs', type=float, default=HEADER_SKIP_RATIO,
                        help='Fraction of image height to skip at top (for header)')
    parser.add_argument('--invert', action='store_true',
                        help='Force inverted polarity (dark pages on light card). '
                             'Default: auto-detected per card.')
    parser.add_argument('--no-invert', action='store_true',
                        help='Force normal polarity (bright pages on dark card), '
                             'disabling auto-detection.')
    parser.add_argument('--padding', '-p', type=float, default=None,
                        help='Crop margin around detected pages (pixels, or fraction of '
                             f'the median page if 0-1). Default: {DEFAULT_PADDING_RATIO} '
                             'of the median page.')
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
    parser.add_argument('--no-split', action='store_true',
                        help='Do not split merged detections at projection '
                             'valleys (splitting is the default).')
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
    parser.add_argument('--manual-boxes', default=None, metavar='CSV',
                        help='Cut exactly these boxes instead of detecting '
                             'anything: one line per page, x,y,w,h in full '
                             'resolution, no header line. The FIRST line is '
                             'the header and becomes page_000. Detection, '
                             'repair, snap and every guard are skipped, and '
                             'no crop margin is applied - for cards the '
                             'operator has laid out by hand after the '
                             'automatic run failed (C24).')
    parser.add_argument('--no-header-page', action='store_true',
                        help='Do not write the masked header band as '
                             'pages/page_000.tif (contract C11).')
    parser.add_argument('--header-page', action='store_true',
                        help='Deprecated no-op: writing the header is the default '
                             'as of 2026-08-23. Still accepted so existing callers '
                             'do not fail — argparse exits 2 on an unknown flag, '
                             'which collides with EXIT_NO_PAGES.')
    parser.add_argument('--background-first', action='store_true',
                        help='Binarize by selecting the BACKGROUND (jacket) '
                             'and taking the complement: foreground is any '
                             'deviation from the local jacket level, in '
                             'either direction. Catches faded/washed pages '
                             'a global threshold cannot see. Flagged until '
                             'field-validated against the default mode.')
    parser.add_argument('--anon-viz', action='store_true',
                        help='Also write _debug/anon_viz.jpg: detected blobs as '
                             'solid black/white silhouettes with boxes and the '
                             'quality banner, but no readable content. Safe to '
                             'take off an air-gapped machine for diagnostics.')
    parser.add_argument('--debug', action='store_true',
                        help='Also write the binary TIFF and box overlay to <card>/_debug/. '
                             'Off by default: the OCR app reads loose image files in the '
                             'card folder as pages.')
    args = parser.parse_args()
    # One line per run so every captured log documents the environment it ran
    # in - environment drift on the offline machine was once a suspect - AND
    # which code and mode produced it (steg 1, 2026-09-08).
    print(f"Env: python {platform.python_version()}, numpy {np.__version__}, "
          f"opencv {cv2.__version__}, pyvips {pyvips.__version__}, "
          f"code {code_version()}, mode {run_mode(args)}")

    if not args.input:
        print("ERROR: --input is required", file=sys.stderr)
        return 1

    if args.invert and args.no_invert:
        print("ERROR: --invert and --no-invert are mutually exclusive",
              file=sys.stderr)
        return 1

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

    if args.manual_boxes:
        # The operator has drawn the pages; cut exactly those and nothing
        # else (C24). No detection, no repair, no snap, no guard, no
        # quality score - that is the whole point of manual mode.
        try:
            manual = read_manual_boxes(args.manual_boxes)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if len(manual) < 2:
            print(f"ERROR: {args.manual_boxes} holds {len(manual)} box(es). "
                  "The first is the header (page_000) and at least one page "
                  "must follow, so two lines are the minimum.",
                  file=sys.stderr)
            return 1
        problems = validate_manual_boxes(manual, original_width,
                                         original_height)
        if problems:
            print(f"ERROR: {len(problems)} box(es) cannot be cut - nothing "
                  "written:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        for warning in validate_manual_boxes(manual, original_width,
                                             original_height, warn=True):
            print(f"WARNING: {warning} (cutting them as drawn)",
                  file=sys.stderr)
        pages = manual[1:]
        print(f"MANUAL boxes: {len(pages)} pages placed by operator, "
              "no padding (first box is the header, cut as page_000)")
        print("\n=== PAGE COORDINATES (full resolution) ===")
        print("Page#, X, Y, Width, Height")
        for i, (x, y, w, h) in enumerate(manual, 0):
            print(f"{i:3d}, {x}, {y}, {w}, {h}")
        csv_path = out_dir / "page_coordinates.csv"
        with open(csv_path, 'w') as f:
            f.write("# Manual boxes placed by operator - no quality score\n")
            f.write("Page#,X,Y,Width,Height\n")
            for i, (x, y, w, h) in enumerate(manual, 0):
                f.write(f"{i},{x},{y},{w},{h}\n")
        print(f"Coordinates saved to {csv_path}")
        finish_card(manual, input_file, input_path, out_dir, original_width,
                    original_height, args, padding=0, header=False,
                    start_number=0)
        print("\nDone!")
        return 0

    # Compute illumination field + Otsu threshold from a thumbnail
    print("Computing illumination field and Otsu threshold from thumbnail...")
    thumb_vips = gray.resize(min(1.0, ILLUM_THUMB_WIDTH / gray.width))
    thumb = np.ndarray(buffer=thumb_vips.write_to_memory(), dtype=np.uint8,
                       shape=[thumb_vips.height, thumb_vips.width])
    illum_field, illum_norm, otsu_thresh, reclassified = illumination_plan(thumb)
    first_pass_thresh = otsu_thresh
    if otsu_override is not None:
        otsu_thresh = otsu_override
        print(f"Step 2 threshold in use: {otsu_thresh:.0f} "
              f"(first pass had {first_pass_thresh:.0f})")
    print(f"Otsu threshold: {otsu_thresh:.0f} (illumination-flattened; "
          f"flattening re-classified {reclassified:.1%} of thumbnail pixels)")
    # The measurement goes in the log on EVERY run (steg 5B): it is how the
    # drift is followed across report folders, and it is what re-calibrated
    # this threshold in the first place. Only an outlier is a warning -
    # 0.8-1.9 % is the normal range for these panoramas (16 field cards,
    # both modes), so warning below that was noise on 100 % of cards.
    print(f"Illumination re-classified {reclassified:.1%} of the thumbnail "
          f"(warns above {ILLUM_WARN_SHARE:.0%})")
    illum_note = None
    if reclassified > ILLUM_WARN_SHARE:
        illum_note = f"UNEVEN ILLUMINATION ({reclassified:.0%} re-classified)"
        print(f"WARNING: uneven illumination - flattening re-classified "
              f"{reclassified:.1%} of pixels, above the {ILLUM_WARN_SHARE:.0%} "
              "outlier threshold (mottled panorama). Detection compensates; "
              "the source may want re-stitching review.")

    # A (nearly) uniform surface gives Otsu threshold 0, which makes the ENTIRE
    # image foreground and the whole card come back as one giant "page" -
    # something wrong that looks normal. Not a card; fail like the no-pages
    # case rather than guess around it.
    degenerate = None
    if otsu_thresh <= 0:
        degenerate = (f"degenerate Otsu threshold {otsu_thresh} "
                      "(near-uniform image, everything is foreground)")

    # Apply the threshold to the full image - still full-res-threshold-
    # then-resize: the detect pass depends on that order (see the
    # resize/threshold duality note in HANDOFF).
    def full_res_surface(values):
        fh, fw = values.shape
        img = pyvips.Image.new_from_memory(
            np.ascontiguousarray(values.astype(np.float32)).tobytes(),
            fw, fh, 1, 'float')
        img = img.resize(original_width / fw, vscale=original_height / fh,
                         kernel='linear')
        if (img.width, img.height) != (original_width, original_height):
            img = img.embed(0, 0, max(img.width, original_width),
                            max(img.height, original_height),
                            extend='copy').crop(0, 0, original_width,
                                                original_height)
        return img

    if args.background_first:
        # Foreground = deviation from the local jacket level, either way.
        print("Background-first mode: foreground = deviation beyond "
              f"{BG_BAND_RATIO:.0%} of the local jacket level")
        level_img = full_res_surface(illum_field)
        binary = (gray - level_img).abs() > (level_img * BG_BAND_RATIO)
    else:
        print("Applying illumination-corrected threshold to full image...")
        binary = gray >= full_res_surface(
            illum_field * (otsu_thresh / illum_norm))

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
    # pyvips relational ops already yield 0/255 (NOT 0/1 - multiplying by 255
    # here wrapped 255 to 1 in uint8 and silently broke --invert); the resize
    # interpolates edge pixels, so re-binarize at the midpoint.
    binary_img = ((binary_img > 127) * 255).astype(np.uint8)

    # Save full-res binary TIFF for reference (optional)
    if args.debug:
        print(f"Saving 1-bit TIFF to {temp_tiff}...")
        binary.write_to_file(str(temp_tiff), compression='lzw', bigtiff=True)

    # A detect-scale copy of the GRAYTONE for the per-cell measurement
    # (steg 8A). The binary is what failed on the faded cards, so it cannot
    # be its own witness - the evidence is measured on the grey.
    gray_small_vips = gray.resize(detect_scale)
    gray_small = np.ndarray(
        buffer=gray_small_vips.write_to_memory(), dtype=np.uint8,
        shape=[gray_small_vips.height, gray_small_vips.width]).copy()

    # Free memory
    del image, gray, binary, binary_small, gray_small_vips

    # === STEP 2: Find contours in the binary image ===
    print("Finding contours on downsampled image...")

    # Threshold polarity. The two known card types are opposite (Yamaha-type:
    # bright pages on dark card; journal jackets: dark pages on light card)
    # and the app sends no flag, so the default is per-card auto-detection -
    # announced LOUDLY, because a silent wrong guess looks like a normal run.
    min_w = int(original_width * detect_scale * MIN_PAGE_WIDTH_RATIO)
    min_h = int(original_height * detect_scale * MIN_PAGE_HEIGHT_RATIO)
    header_skip_px_small = int(original_height * detect_scale * args.header_skip)

    if args.background_first:
        # The deviation mask IS the content, regardless of which side of
        # the jacket level it sits on - polarity does not exist here.
        do_invert = False
        print("Polarity: not applicable (background-first deviation mask)")
    elif args.no_invert:
        do_invert = False
    elif args.invert:
        do_invert = True
        print("Inverting binary image (forced by --invert)...")
    else:
        do_invert = autodetect_inversion(
            binary_img, header_skip_px_small, min_w, min_h)
        if do_invert:
            print("Auto-detected polarity: INVERTING "
                  "(dark pages on a light card)")
    if do_invert:
        binary_img = cv2.bitwise_not(binary_img)

    # Second degenerate-threshold symptom: a real Otsu value but ~everything
    # above it (blank bright scan). Same one-giant-page failure as threshold 0.
    foreground_share = float(np.count_nonzero(binary_img)) / binary_img.size
    if degenerate is None and foreground_share > FOREGROUND_SANE_MAX:
        degenerate = (f"foreground is {foreground_share:.1%} of the image "
                      f"(sane maximum {FOREGROUND_SANE_MAX:.0%}; "
                      "blank or washed-out scan, not a card)")

    # === STEP 3: Detect, filter and sort bounding boxes ===
    print(f"Skipping top {header_skip_px_small} pixels in downsampled image (header region)")
    print(f"Detecting pages (erosion kernel={DETECT_ERODE_KERNEL})...")
    structure_rows = []   # detect-scale y-runs of the deleted stripes
    border_share_out = []
    boxes, filtered_contours, binary_img, det_log, small_witnesses = \
        detect_page_boxes(binary_img, header_skip_px_small, min_w, min_h,
                          collect_witnesses=True,
                          structure_rows_out=structure_rows,
                          report_scale=1 / detect_scale,
                          border_share_out=border_share_out)
    border_share = border_share_out[0] if border_share_out else 0.0
    for line in det_log:
        print(line)
    # The stripes are the row boundaries (steg 2): full-res in the report so
    # the row slots can be checked against the field cards.
    stripes_fullres = [(int(a / detect_scale), int(b / detect_scale))
                       for (a, b) in structure_rows]
    if stripes_fullres:
        print("Structure rows (full-res y): "
              + ", ".join(f"{a}-{b}" for a, b in stripes_fullres))

    print(f"Found {len(boxes)} potential pages")

    # The staircase (C9, steg 7): when the first threshold demonstrably did
    # not find the card and what it did find is essentially all frame, try
    # ONE second threshold with the frame and the known structure taken out
    # of the histogram - then run the whole pass again on its result. Step
    # two earns nothing: the card must pass every guard on its own.
    def try_step_two(why, require_border=True):
        if step2 or otsu_override is not None or args.background_first:
            return None
        if require_border and border_share <= STEP2_BORDER_TRIGGER:
            return None
        header_px_thumb = int(thumb.shape[0] * args.header_skip)
        mask = frame_mask(thumb, first_pass_thresh, header_px_thumb)
        flat = np.clip(thumb.astype(np.float32)
                       * (illum_norm / cv2.resize(illum_field,
                                                  (thumb.shape[1],
                                                   thumb.shape[0]),
                                                  interpolation=cv2.INTER_LINEAR)),
                       0, 255).astype(np.uint8)
        new_thresh = otsu_excluding(flat, mask)
        print(f"\nStep 2 threshold: trigger {why}, border "
              f"{border_share:.0%} (over {STEP2_BORDER_TRIGGER:.0%}), "
              f"otsu {first_pass_thresh:.0f} -> "
              f"{'none' if new_thresh is None else format(new_thresh, '.0f')}")
        if new_thresh is None or abs(new_thresh - first_pass_thresh) < 1:
            print("  Step 2 found no different threshold - not retrying",
                  file=sys.stderr)
            return None
        return new_thresh

    if not boxes and not degenerate:
        retry = try_step_two("0 detections")
        if retry is not None:
            return main(otsu_override=retry, step2=True,
                        step1_border=border_share)

    # Fail loudly on a total detection failure. Writing an empty card folder
    # would be worse than useless: the OCR app skips empty folders silently, so
    # the card would vanish from the queue with no error anywhere.
    if degenerate or not boxes:
        # C9: after a step-two attempt the card must be told the reason it
        # actually had. "no pages detected" would blame the card for a
        # threshold's mistake - the staircase ran and did not recover it.
        if not boxes and step2 and step1_reasons:
            # Step two made it WORSE: the first pass found pages and failed
            # a guard, the second found nothing. Report the diagnosis the
            # card actually had - losing it would trade a real reason for a
            # threshold's excuse (steg 9E).
            print(f"\nStep 2 (threshold {otsu_thresh:.0f}) found no pages "
                  "at all - keeping the first pass's diagnosis",
                  file=sys.stderr)
            for r in step1_reasons:
                print(f"\nERROR: {r}", file=sys.stderr)
            print(f"\nERROR: suspected split pages in {input_file} - "
                  "not extracting.", file=sys.stderr)
            print("  No _done sentinel written — this card will not be "
                  "offered for import.", file=sys.stderr)
            if not args.skip_extraction:
                moved = move_without_clobber(input_path,
                                             input_path.parent / "error")
                print(f"  Source scan moved to {moved}", file=sys.stderr)
            else:
                print("  Source left in place (--skip-extraction is "
                      "inspection-only).", file=sys.stderr)
            return EXIT_SUSPECT_FRAGMENTS
        if not boxes and step2:
            reason = (f"threshold found only the frame; re-threshold failed "
                      f"(border {step1_border:.0%} -> {border_share:.0%}, "
                      f"second threshold {otsu_thresh:.0f} found no pages "
                      "either)")
        else:
            reason = degenerate or "no pages detected"
        print(f"\nERROR: {reason} in {input_file}", file=sys.stderr)
        print("  No _done sentinel written — this card will not be offered for import.",
              file=sys.stderr)
        # A no-pages failure is exactly when the picture matters most: write
        # the thresholded view so the operator can SEE what detection saw.
        fail_scale = min(1.0, 2000 / max(binary_img.shape[1], binary_img.shape[0]))
        fail_viz = cv2.resize(binary_img, None, fx=fail_scale, fy=fail_scale)
        cv2.imwrite(str(viz_path), fail_viz)
        print(f"  Detection view saved to {viz_path}", file=sys.stderr)
        # The anonymized view is written on failure too - failing cards are
        # exactly the ones that must be inspectable across the air gap. The
        # silhouettes of whatever survived detection (often nothing: a black
        # frame) plus the failure reason still carry no readable content.
        if args.anon_viz:
            fail_radius = erosion_radius(DETECT_ERODE_KERNEL,
                                         DETECT_ERODE_ITERATIONS)
            anon_mask = make_anon_mask(binary_img.shape, filtered_contours,
                                       fail_radius)
            fail_boxes = [(int(x / detect_scale), int(y / detect_scale),
                           int(w / detect_scale), int(h / detect_scale))
                          for (x, y, w, h) in boxes]
            anon_viz = render_anon_viz(anon_mask, fail_boxes, detect_scale,
                                       f"FAILED: {reason}"
                                       + (f"  |  {illum_note}" if illum_note
                                          else ""),
                                       (0, 0, 200))
            anon_path = debug_dir / "anon_viz.jpg"
            cv2.imwrite(str(anon_path), anon_viz)
            print(f"  Anonymized view saved to {anon_path}", file=sys.stderr)
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
    boxes, filtered_contours, dropped = drop_band_detections(
        boxes, filtered_contours, BAND_RATIO)
    if dropped:
        scale_up = 1.0 / detect_scale
        print(f"Rejected {len(dropped)} band-shaped detection(s) (not pages):")
        for (x, y, w, h) in dropped:
            print(f"  at ({int(x * scale_up)}, {int(y * scale_up)}) "
                  f"size {int(w * scale_up)} x {int(h * scale_up)} full-res")
        print(f"{len(boxes)} pages remain")

    # === STEP 4: Scale coordinates back to original size ===
    scale_back = 1.0 / detect_scale
    boxes_fullres = []
    for (x, y, w, h) in boxes:
        boxes_fullres.append((
            int(x * scale_back),
            int(y * scale_back),
            int(w * scale_back),
            int(h * scale_back)
        ))

    # Machine-readable geometry for the field regression (steg 5C).
    # Coordinates only - no content - so the reports stay safe to carry off
    # the air-gapped machine. RAW is what detection found BEFORE the split
    # pass: it is diagnosis, not a replay input, because splitting reads the
    # panorama and no coordinate-only replay can reproduce it. The
    # difference between RAW and PRE-REPAIR is exactly what splitting did -
    # which is where card 098 lied (sixteen single pages "split into 2-4").
    print("RAW detections (full-res x,y,w,h): "
          + "; ".join(f"{x},{y},{w},{h}" for x, y, w, h in boxes_fullres))

    # === STEP 4a: Split merged detections at projection valleys ===
    # Weak edges fuse touching pages into one detection; the gap between real
    # pages is a projection valley at scan scale. Scanned per box against the
    # source file (like refine), split boxes replace their merge.
    split_count = 0
    if not args.no_split:
        min_w_full = int(original_width * MIN_PAGE_WIDTH_RATIO)
        min_h_full = int(original_height * MIN_PAGE_HEIGHT_RATIO)

        def _split_one(box):
            # Same view as the main pass: local scalars from the same field.
            local_level = illumination_local_threshold(
                illum_norm, illum_field, illum_norm, box,
                original_width, original_height)
            if args.background_first:
                return split_box_by_projection(
                    input_file, box, 0, False, min_w_full, min_h_full,
                    deviation=(local_level, BG_BAND_RATIO * local_level))
            local_thresh = illumination_local_threshold(
                otsu_thresh, illum_field, illum_norm, box,
                original_width, original_height)
            return split_box_by_projection(
                input_file, box, local_thresh, do_invert, min_w_full, min_h_full)

        # Only boxes that could actually hold two pages are scanned: the
        # rest are single pages, and a valley inside one is washed content,
        # not a gutter (steg 5B). The basis is the FORMAT size, never the
        # median of a box list the fragments dominate - that median is what
        # made card 098 split sixteen single pages. resolve_page_size takes
        # the prior whenever any detection matches it (a whole page always
        # does), so a fragment-heavy card cannot shrink the gate; only a
        # card with no format-sized detection at all falls to the per-card
        # estimate, and then there is no constant to lean on. Say which.
        split_w, split_h, split_basis = resolve_page_size(boxes_fullres)
        if split_basis and split_basis.startswith("Page-size prior"):
            print(f"Split gate on per-card estimate {split_w} x {split_h} "
                  "(no detection matches the format prior)")
        pieces = [[b] for b in boxes_fullres]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_split_one, b): i
                       for i, b in enumerate(boxes_fullres)
                       if can_hold_two_pages(b, split_w, split_h)}
            for future in as_completed(futures):
                pieces[futures[future]] = future.result()

        split_boxes = []
        for original, parts in zip(boxes_fullres, pieces):
            if len(parts) > 1:
                split_count += len(parts) - 1
                x, y, w, h = original
                print(f"Split merged detection at ({x}, {y}) {w} x {h} "
                      f"into {len(parts)} pages")
            split_boxes.extend(parts)
        boxes_fullres = split_boxes

    # Warn loudly about kept boxes that STILL look like several pages fused
    # into one after the split attempt (no usable valley). They are kept -
    # dropping loses content - but the operator must see it in the log, not
    # just as a wide box in the viz.
    if len(boxes_fullres) >= 4:
        warn_w, warn_h, _ = resolve_page_size(boxes_fullres)   # format, not median
        for (x, y, w, h), fused in suspected_merged_boxes(boxes_fullres,
                                                          warn_w, warn_h):
            print(f"WARNING: suspected merged pages (~{fused} fused): "
                  f"at ({x}, {y}) size {w} x {h} full-res")

    # === Compute card quality score ===
    # After a split the detect-scale contours no longer correspond to the
    # boxes; the shape component then falls back to neutral.
    quality = compute_card_quality(
        boxes_fullres, filtered_contours if split_count == 0 else None)

    # Sort based on reading order
    if args.order == 'columns':
        print("Sorting by columns (left-to-right, then top-to-bottom within each column)")
        boxes_fullres = sort_boxes_by_columns(boxes_fullres)
    else:
        print("Sorting by rows (top-to-bottom, then left-to-right within each row)")
        boxes_fullres = sort_boxes_by_rows(boxes_fullres)

    # === STEP 4b: Optionally refine positions at higher local resolution ===
    if args.refine:
        print("\n=== REFINING PAGE POSITIONS ===")
        print("Re-detecting each page locally at 20% resolution...")

        header_skip_fullres = int(original_height * args.header_skip)

        def _refine_one(box):
            local_level = illumination_local_threshold(
                illum_norm, illum_field, illum_norm, box,
                original_width, original_height)
            if args.background_first:
                return refine_box_local(
                    input_file, box, 0, original_width, original_height,
                    header_skip_px=header_skip_fullres,
                    deviation=(local_level, BG_BAND_RATIO * local_level))
            local_thresh = illumination_local_threshold(
                otsu_thresh, illum_field, illum_norm, box,
                original_width, original_height)
            return refine_box_local(
                input_file, box, local_thresh,
                original_width, original_height,
                invert=do_invert, header_skip_px=header_skip_fullres)

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

    # Phase 2 and the snap: one coordinate-only chain, extracted 2026-09-08
    # so a report's PRE-REPAIR line can be replayed through exactly this
    # code. main keeps the printing and the exit decision.
    witnesses_fullres = [
        (int(x / detect_scale), int(y / detect_scale),
         int(w / detect_scale), int(h / detect_scale))
        for (x, y, w, h) in small_witnesses]
    # PRE-REPAIR is the replay input: the last point where everything below
    # is pure geometry. Feed these three lines (with the structure rows)
    # back through repair_and_snap and the result is what shipped. With
    # --refine it is not bit-exact - refine reads the image too - but
    # production does not use it.
    print("PRE-REPAIR detections (full-res x,y,w,h): "
          + "; ".join(f"{x},{y},{w},{h}" for x, y, w, h in boxes_fullres))
    print("RAW witnesses (full-res x,y,w,h): "
          + "; ".join(f"{x},{y},{w},{h}" for x, y, w, h in witnesses_fullres))
    chain = repair_and_snap(boxes_fullres, witnesses_fullres,
                            stripes_fullres, original_width, original_height,
                            header_px=int(original_height * args.header_skip))
    for stream, text in chain.output:
        print(text, file=sys.stdout if stream == 1 else sys.stderr)
    boxes_fullres = chain.boxes
    geo_indices = chain.repaired
    repaired_count = len(geo_indices)
    fragment_groups = chain.fragment_groups
    fragment_indices = {i for group in fragment_groups for i in group}
    refused_groups = chain.refused_groups
    snap_refused = chain.snap_refused
    geo_overload = chain.geo_overload
    card_refusals = chain.card_refusals
    if chain.evidence_refused or chain.layout_refused:
        # A layout refusal needs no border condition (steg 9E): an
        # impossible layout is itself evidence that the threshold is wrong,
        # and card 647 splintered at a border share of only 26 %.
        retry = try_step_two(
            "evidence guard refused the card" if chain.evidence_refused
            else "layout invariant broken",
            require_border=chain.evidence_refused)
        if retry is not None:
            return main(otsu_override=retry, step2=True,
                        step1_border=border_share,
                        step1_reasons=tuple(chain.card_refusals))
    if chain.quality is not None:
        quality = chain.quality

    # Step two must prove itself (C9): the border share has to at least
    # halve against the first pass, and the page size must come from the
    # format prior rather than a per-card estimate. Otherwise the card is
    # refused with the reason it actually had.
    if step2:
        # Step two proves itself by the card passing every guard on its own
        # (steg 9C) - not by a number about the frame. The border share is
        # a FRACTION of total foreground, so a card with few pages reads
        # high however well the threshold worked: 612130000609_00036 was
        # refused at border 100% -> 61% while shipping 17 clean pages in
        # 12+5 at quality 94.2. Border is logged, and decides nothing.
        first_border = step1_border if step1_border is not None else 1.0
        _pw, _ph, size_note = resolve_page_size(boxes_fullres)
        off_prior = bool(size_note and size_note.startswith("Page-size prior"))
        if off_prior:
            card_refusals = card_refusals + [
                "threshold found only the frame; re-threshold failed "
                f"(border {first_border:.0%} -> {border_share:.0%}; the page "
                "size still does not match the format prior)"]
            print(f"\nERROR: {card_refusals[-1]}", file=sys.stderr)
        elif card_refusals or fragment_groups or refused_groups or snap_refused:
            print(f"\nStep 2 ran (border {first_border:.0%} -> "
                  f"{border_share:.0%}) but the card fails a guard on its "
                  "own - see the reason above", file=sys.stderr)
        else:
            print(f"\nStep 2 proved itself: border {first_border:.0%} -> "
                  f"{border_share:.0%}, {len(boxes_fullres)} pages on the "
                  "format prior, every guard passed")

    # Per-cell evidence (steg 8A): measurement only, nothing decides on it.
    # One machine-readable line per cell, empty cells included, so the
    # calibration in 8B can be scripted straight off the report folders.
    # ...and because it decides nothing, it must not be able to decide
    # anything by crashing either: a measurement that fails takes the card
    # down with it otherwise.
    try:
        cell_pw, cell_ph, _ = resolve_page_size(boxes_fullres)
        omitted_cells = []
        for cell in card_cells(boxes_fullres, cell_pw, cell_ph,
                               original_width, omitted_out=omitted_cells):
            local = illumination_local_threshold(
                otsu_thresh, illum_field, illum_norm,
                (cell["x"], cell["y"], cell_pw, cell_ph),
                original_width, original_height)
            fg, edge = cell_evidence(gray_small, cell, cell_pw, cell_ph,
                                     detect_scale, local,
                                     dark_pages=do_invert)
            print(f"CELL row={cell['row']} x={cell['x']} y={cell['y']} "
                  f"page={cell['page']} fg={fg:.3f} edge={edge:.3f}")
        if omitted_cells:
            print(f"Cell margin: {omitted_cells[0]} cell(s) omitted "
                  "outside the raster the rows span - card margin, not "
                  "evidence (steg 9F)")
    except Exception as exc:                       # measurement only - never
        print(f"CELL measurement failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)                     # ...a reason to fail

    # Coverage guard: did the boxes cover what the threshold saw? The one
    # signal that survives any upstream mistake (row-banding collapse put a
    # whole page row outside every box on card 612130000036 - at quality
    # 100). Mandatory in every mode.
    coverage_note = None
    boxes_detect = [(x * detect_scale, y * detect_scale,
                     w * detect_scale, h * detect_scale)
                    for (x, y, w, h) in boxes_fullres]
    outside_share = foreground_outside_boxes(binary_img, boxes_detect)
    if outside_share > COVERAGE_WARN_SHARE:
        coverage_note = (f"COVERAGE: {outside_share:.0%} of foreground "
                         "outside all boxes")
        cap = round(100.0 * (1.0 - outside_share), 1)
        print(f"\nWARNING: {outside_share:.0%} of the foreground mass lies "
              f"outside every page box - content the boxes do not cover. "
              f"Quality capped at {cap}.", file=sys.stderr)
        if quality['total'] > cap:
            quality['total'] = cap

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
        if (i - 1) in fragment_indices:
            box_color = FRAGMENT_MARK_COLOR
        elif (i - 1) in geo_indices:
            box_color = GEOMETRY_MARK_COLOR
        else:
            box_color = (0, 255, 0)
        cv2.rectangle(viz, (sx, sy), (sx + sw, sy + sh), box_color, 2)
        cv2.putText(viz, str(i), (sx + 5, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add quality score banner at the top
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
    if do_invert:
        label += "  |  inverted" + ("" if args.invert else " (auto)")
    if illum_note:
        label += f"  |  {illum_note}"
    if repaired_count:
        label += f"  |  {repaired_count} geometry-completed"
    if coverage_note:
        label += f"  |  {coverage_note}"
    if (fragment_groups or refused_groups or geo_overload
            or snap_refused or card_refusals):
        n_suspect = (len(fragment_groups) + len(refused_groups)
                     + len(snap_refused) + len(card_refusals))
        label = f"SUSPECT FRAGMENTS ({n_suspect} group(s))  |  " + label
        banner_color = (0, 0, 200)
    cv2.putText(banner, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2)
    viz = np.vstack([banner, viz])

    cv2.imwrite(str(viz_path), viz)
    print(f"Saved {viz_path}")

    # Anonymized view for air-gapped diagnostics: solid silhouettes of the
    # detected blobs (content filled shut, seam gaps preserved) with the same
    # boxes and banner. The normal visualization above shows readable journal
    # content and must stay on the machine; this one may leave it.
    if args.anon_viz:
        anon_mask = make_anon_mask(binary_img.shape, filtered_contours,
                                   detect_radius)
        anon_viz = render_anon_viz(anon_mask, boxes_fullres, detect_scale,
                                   label, banner_color,
                                   fragment_indices=fragment_indices,
                                   repaired_indices=geo_indices)
        anon_path = debug_dir / "anon_viz.jpg"
        cv2.imwrite(str(anon_path), anon_viz)
        print(f"Saved {anon_path} (anonymized)")

    # Guard verdict, after both visualizations exist with the suspects
    # marked: irreconcilable chains, REFUSED merges (a sub-band refusal is
    # invisible to the re-run guard, whose lower bound is 0.8), or a card
    # geometry had to repair more of than the limit. The source goes to
    # error/ for review, like the no-pages failure.
    if (fragment_groups or refused_groups or geo_overload
            or snap_refused or card_refusals):
        print(f"\nERROR: suspected split pages in {input_file} - "
              "not extracting.", file=sys.stderr)
        print("  No _done sentinel written — this card will not be offered "
              "for import.", file=sys.stderr)
        if not args.skip_extraction:
            moved = move_without_clobber(input_path, input_path.parent / "error")
            print(f"  Source scan moved to {moved}", file=sys.stderr)
        else:
            print("  Source left in place (--skip-extraction is "
                  "inspection-only).", file=sys.stderr)
        return EXIT_SUSPECT_FRAGMENTS

    finish_card(boxes_fullres, input_file, input_path, out_dir,
                original_width, original_height, args)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
