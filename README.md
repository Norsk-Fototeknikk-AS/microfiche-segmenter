# microfiche-segmenter

Cuts a stitched microfiche panorama into one image file per page.

Input is a gigapixel panorama produced by the PTGui runner (one physical fiche
card = one panorama, stitched from 16 tiles). Output is a *card folder* that the
OCR-Pipeline app watches and imports on operator command.

```
segment_microfiche.py  --input <panorama> --output <card folder>
```

The whole job is: threshold the card, find the page rectangles, write one crop
per page plus their coordinates, then mark the folder finished.

---

## Contracts — DO NOT BREAK THESE

Everything in this section is depended on by another program, by an agreement
made outside this repo, or by physics of the image pipeline. **The test suite
passing is not sufficient evidence that a change here is safe.** These rules
cannot be re-derived from the code, and several of them look like dead weight
until you know why they exist. If you are tempted to simplify one of them,
read the reason first.

### C1. Erosion must be compensated

Detection runs on a 10 % downscale and erodes the binary image (7×7 kernel,
2 iterations) to pull touching pages apart. Erosion shrinks every blob by
`(kernel // 2) * iterations` = **6 px at detect scale = 60 px at full
resolution, on all four sides**.

`expand_boxes()` adds exactly that back. Delete it and every page silently
loses ~60 px of its own edge — text at the page margin gets cut. Nothing in the
output looks wrong: the crops are still page-shaped, the quality score still
reads GOOD, and the tests that don't measure geometry still pass. This bug
shipped once already and was only found by measuring crops against the source.

`refine_box_local()` has its own erosion (3×3, 2 iterations, at 20 % local
scale = 10 px full-res) and its own compensation. Both must stay, or the two
detection passes disagree by ~50 px.

### C2. `_done` is the import signal, and its ordering is load-bearing

The OCR app treats the presence of `_done` in a card folder as "fully written,
safe to import".

- Written **last**, after every page file is on disk.
- Written **atomically** (temp file + `os.replace`), never built in place.
- Deleted **first** at startup, *before* `pages/` is cleared.

That last ordering matters: while `_done` exists the app considers the card
importable, so it must be gone before anything starts rewriting the folder. A
re-run that clears pages while a stale `_done` sits there will hand the app a
half-written card. This was measured happening — the sentinel stayed visible for
the full 10 s of a rewrite.

Without `_done` the app falls back to "120 s with no file changes", which is
fragile. Always write it on success.

### C3. Stale pages must be purged at startup

`pages/` is emptied before a run writes to it. If a previous run found 147 pages
and this one finds 140, `page_141..147` would otherwise survive and be imported
as real pages of the journal. Same-name overwriting is not enough.

### C4. `--skip-extraction` must never destroy anything

It is inspection mode: it writes coordinates only. It must not clear `pages/`
and must not remove `_done`, because the card on disk may be a finished, valid
card. Running it must be safe on any folder at any time.

### C5. The card folder contains no loose image files

A card folder holds exactly:

```
<card>/
    _done
    page_coordinates.csv
    pages/page_000.tif    ← the card header (C11)
    pages/page_001.tif …
```

The OCR app has a fallback that reads image files sitting *directly* in the card
folder as pages, used when `pages/` is missing. Anything else image-shaped left
at that level can therefore be imported as a page. Debug artifacts live in
`<card>/_debug/`, never at card level: the box overlay `visualization.jpg` is
written there on every run (2026-09-03) as the operator's ground truth for what
was detected and in which order; the full-res binary only with `--debug`.

### C6. Output naming is the system boundary

- The card folder name is the **input file stem, verbatim**. We do not rename.
- It must match `^(\d{12})(?:\D|$)` — 12-digit fanearkID, then a non-digit or
  end of string. A 13th digit makes the app skip the folder *silently*.
- The first integer *after* the 12 digits is the card's order within a journal.
  One journal can span several cards; PTGui names them from the last tile, so
  card 1 = `<fid>_00016`, card 2 = `<fid>_00032`.
- Page order is the **last integer in the filename stem**, sorted naturally.
  `page_%03d` zero-padding is safe and wanted.

### C7. TIFF end to end

Page crops are TIFF (LZW). Decided 2026-08-12: the workflow stays TIFF until
final packaging, because a JPG crop here re-encodes already-stitched pixels and
the archival TIFF downstream inherits the artifacts. `--format jpg` exists but
must not be used for pipeline output.

### C8. First CSV line carries the quality score

`page_coordinates.csv` line 1 must contain `Card Quality: N/100`. The app
parses it for its quality panel; the worst card's score represents the journal.
If the line or the file is missing the app degrades silently — no error, just a
missing number.

### C9. Failure must be loud

A card that detects zero pages must **not** produce a card folder. The app
ignores empty folders silently, so an empty folder is invisible, not an error
signal. On zero detections: no `_done`, message on stderr, the source panorama
is moved to `<input dir>/error/`, the empty card folder is removed, exit 2.

A degenerate threshold takes the same path (2026-09-03). A (nearly) uniform
image gives Otsu threshold 0, and a blank bright scan gives a real threshold
with ~everything above it (`FOREGROUND_SANE_MAX`, 97 %); both would otherwise
emit the whole card as one giant "page" — something wrong that looks normal.

**Re-thresholding, amended 2026-09-08 (Trond's decision).** This contract
used to end "No re-thresholding heuristics: no real panorama has failed this
way yet, so the failure is reported, not guessed around." That was written
when no card had failed that way. In the 88-card production run **six did**,
identically: 494_00012, 494_00036, 609_00012, 609_00024, 609_00036 and
623_00024 all detected **zero** pages with the bottom band as their only
structure run and 99–100 % of their foreground touching the image border.

The mechanism is measured, not guessed. Grey levels on the committed fasit
cards: **frame 2, page 33, stripe 55, jacket 226**. A page sits next to the
*frame*, not next to the jacket, so a healthy card has one dark cluster
{2…55} against the jacket, Otsu lands at ~131 in the wide gap, and
`clear_border_connected` then removes frame and stripes and leaves the
pages — border share 12–24 %. On an over-exposed card the page level has
risen toward the jacket, the only dark mass left is the frame, and Otsu
splits *frame against everything else* at 85–111. Every page then falls on
the background side and the foreground that remains **is** the frame.

So re-thresholding is no longer a heuristic guess: it is a second step for a
diagnosed condition. It is allowed **only** as step two of a staircase, and
only under these rules:

- It runs **only** after a measured trigger says the first pass did not find
  the card — never speculatively, never on a card that passed.
- Step two must **prove itself** on its own result. If it cannot, the card is
  refused with the reason it actually had (`threshold found only the frame;
  re-threshold failed`) and both measurements in the message — never
  "no pages detected", which would blame the card for a threshold's mistake.
- Every step-two run is **visible**: its own log line with the trigger value,
  the old and new threshold and what changed, and a mark in `SAMMENDRAG`.
- Failure is still reported, never guessed around. The staircase adds one
  diagnosed step; it does not add a search for something that works.

### C10. Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | unhandled exception (traceback on stderr) |
| 2 | no usable detection (zero pages, or degenerate threshold); source moved to `error/` |
| 3 | suspected page fragments — a page cut horizontally in two detections (stitching seam in the input); source moved to `error/`, no `_done` (see C14) |

`-O` / `--output` names the **card folder**, not the watch root. A caller that
wants `<root>/<fid>/` must pass `-O <root>/<fid>` itself.

### C11. The header band is kept as page zero

`--header-skip` masks the top of the card during detection so the header is not
found as a page. The band written out as `pages/page_000.tif` is card-adaptive
(2026-09-04): it extends from the top down to the first detected page row —
never less than the configured band, capped at twice it. On the real journal
card the typed text sits *below* the 8 % mask line, so a ratio-only crop cut
the date and card index in half; everything above the pages is the header.

**On by default since 2026-08-23**, once the import side could handle it:
`split_scan_dump._journal_pages` separates page zero from the page sequence and
carries it as a sidecar in `00_header/` named after the card, and step 03 reads
it with a field-oriented prompt. `--no-header-page` suppresses it.
`--header-page` is still accepted as a no-op so callers written during the
opt-in period do not break — argparse exits 2 on an unknown flag, which would be
indistinguishable from EXIT_NO_PAGES.

It carries the card's only identifying text — title, part number, date, and an
**"N of M" card index**. That index is an independent witness to which card of a
journal this is. If the operator forgets to chain a second card it receives its
own fanearkID, and two valid 16-tile cards are indistinguishable downstream;
the header saying "2 of 2" is the only known way to catch it. It is also printed
far larger than the same details on the pages themselves, so it is the better
OCR source, and downstream treats it as authoritative on conflict.

Fixed points:

- **Name `page_000`.** The OCR app sorts pages on the last integer in the stem,
  so page zero sorts ahead of page 1 with no special casing. This name is
  load-bearing, not cosmetic: the import step recognises the header *by its
  natural index being 0* and diverts it to a `00_header/` sidecar. Rename it —
  to `header.tif` or anything without a trailing 0 — and it stops being
  recognised, flows on as an ordinary page, and the packaging step numbers it
  from 1 without inspecting what it is. The header would then be delivered in a
  patient's record as page 00001. Packaging has no guard of its own; the name is
  the whole mechanism.
- **Scale 1/16** (`HEADER_PROXY_SCALE`). Not arbitrary: the downstream packaging
  step caps page width at 2480 px, and 2144 px passes through untouched. Larger
  would be scaled back down immediately. All fields stay legible at this size.
- We do **not** write `header.json`. Producing the image is this repo's job;
  interpreting the text is the OCR pipeline's. Do not blur that line.

Consequence when enabled, and it is intentional: `pages/` holds one more file
than the page count we report. `page_coordinates.csv` and the log count
**detected pages**; the file count is **pages + header**. 147 reported, 148
files. Downstream must handle both shapes, since cards segmented before and
after the switch can appear in the same handover.

### C12. Detections that are not page-shaped are rejected

`drop_band_detections()` discards boxes shaped like bands: more than
`BAND_RATIO` (2.5×) the card's median page in one dimension while a thin
*sliver* — at most `BAND_MAX_THICKNESS` (0.75×) of the median — in the other.
Shape, not size (2026-09-03): real journals hold pages of genuinely different
sizes, so mere deviation from the median — the old 40 % `SIZE_TOLERANCE_RATIO`
rule — is not evidence against being a page, and losing a real page costs more
than keeping a blank crop.

The sliver criterion exists because weak page edges can fuse a whole **row**
into one detection: several pages wide but full page height (reported on real
material 2026-09-03). Width cannot tell a merged row from a band — a
full-width band and a fully merged row are equally wide — but thinness can:
the documented real band was 0.44× the median height, and no page is that
flat. A merged detection is first offered to the split pass (below); what
still cannot be split is kept and warned about (`suspected merged pages` in
the log) — dropping loses content.

**Changed 2026-09-08 (steg 4B):** keeping it is no longer the same as
shipping it. A box over `SNAP_IMPOSSIBLE_RATIO` (1.5) pages in either
dimension is *impossible geometry* — several pages fused into one — and now
**fails the card** (exit 3, no `_done`, source to `error/`), while its raw
box stays in the coordinates and the visualizations so the operator can see
what happened. Card 111 in background mode shipped page 1 as 8610×3100 —
four pages in one crop — at exit 0, quality 52.5, and `SAMMENDRAG` said OK;
card 050 had one 4220×2840 box on an otherwise healthy card. Boxes between
the snap exemption (1.25) and this bound still pass through raw with their
warning.

The minimum-size filter only catches specks. The opposite failure is a bright
band along a card edge — far too wide and flat to be a page, far too big to be
noise. Left in, it becomes a blank page mid-sequence and shifts every later page
number by one. Observed for real: a 33190 × 720 strip against a 2040 × 1630
median, which dropped a card from 91.8 GOOD to 58.8 POOR.

Median-relative, so it adapts to any card format. If more than half the boxes
would be dropped the median itself is untrustworthy, so nothing is dropped.
Every rejection is logged — never discard silently.

**This treats a symptom.** The band was real photographed content from the light
table, caused by framing at the copy stand, not by a stitching fault. The filter
is a robustness invariant worth having regardless, but if edge bands start
appearing, the camera framing is where to look — not this code and not the
stitcher.

### C13. A segmented panorama leaves the queue

`Panoramas/` is a work queue. After a card is successfully segmented, its
panorama is moved to `PanoramaArchive/` **beside** that folder, so what remains
in `Panoramas/` is exactly what still needs doing.

Ordering is deliberate: the move happens **after** `_done`, never before. If the
move fails the card is still complete and importable and only needs filing by
hand; the reverse would leave a published card whose source had vanished
mid-write.

- Only on success. A card that detects no pages goes to `error/` instead (C9) and
  is never archived.
- Never under `--skip-extraction`. Inspection must not move the operator's
  source (C4).
- Never overwrites. A name collision means two different scans share a name, so
  the incoming file gets a numeric suffix instead of destroying the resident one.
- `--no-archive` leaves it in place; `--archive-dir` overrides the location.

**Consequence for re-runs:** re-running a card is no longer just re-running the
same command — the panorama has moved, so point at the archived copy or pass
`--no-archive` on the first run. C3 still guarantees the *card folder* is safe to
rewrite; it is the input that is no longer where it was.

Known wart: failed scans go to `error/` *inside* the input folder, while the
archive sits *beside* it. Those two should probably agree. Nobody has decided
which way.

### C14. Fragment chains: geometric completion first, exit 3 for the rest

History: 2026-09-07 the guard BLOCKED every fragmented card (stitching seams,
content possibly missing in the gap — merging would have been fabrication).
2026-09-08 Trond flipped it: stitching is fixed and content is intact; the
remaining defects are material (washed-out patches at jacket brightness,
short documents, half-dark pages). The sheet size is known, so **geometry
overrides the binary** (`complete_geometry`):

- Fragment chains meeting the x-IoU/gap criteria are **merged** into their
  union box — including chains whose union falls BELOW 0.8× the expected
  height (phase 3, field pair 17+27 at 0.79×): those are pages that lost
  height to a defect, and after merging, the extension pass completes them
  to the row's anchors. The upper union bound (1.8×) stays — it is the
  cross-row guard, and field row gaps (~2× the gap criterion) keep short
  results from ever linking across a row boundary. Crops
  are cut from the original graytone, so a washed-out patch keeps whatever
  readable traces it has for OCR. A merge that would invent more than
  `GEOMETRY_MAX_INVENTED_SHARE` (30 %) of the page area is refused — that is
  fabrication, not repair; refused merges fail the card explicitly — a
  refused sub-band chain is invisible to the re-run guard. (Field
  calibration, 28 production groups: every one tiled its union exactly,
  ~0 % invented.)
- **Vertical stripes** merge symmetrically (Trond's override, same day):
  a page split into full-height strips has the outline of one page, and the
  format guarantees uniform page sizes. The union band is tighter
  (`STRIPE_UNION_MIN/MAX` = 0.8–1.2× the expected page width) because the
  whole risk is merging two real neighbour pages — their union is ~2× a page
  wide plus a real gap (field pitch 2180 vs width 2040), which the band
  excludes with wide margin; a test pins that a full row at field pitch
  produces no stripe groups. Two half-width documents in neighbouring
  frames never link either — the frame spacing exceeds the gap criterion.
- A lone **short detection** in a row with ≥2 full-height anchors is
  extended to the row's top edge and median height (pages share their top
  edge within a row; verified on the fasit). Worst case is empty film in
  the crop, so extensions are exempt from the invented-cap.
- Repaired pages are marked **blue** in both visualizations, counted in the
  banner, and logged per page (`N pages geometry-completed`, with invented
  share). Never silent repair.
- The guard is NOT weakened: it re-runs on the repaired geometry, so
  whatever still matches the fragment signature (refused merges included)
  exits 3 as before. And if geometry had to repair more than
  `GEOMETRY_MAX_REPAIR_SHARE` (50 %; field worst case 28 %) of the card's
  pages, the card is genuinely sick — exit 3.

Thresholds are calibrated against RAPPORT-2026-09-08-3 (13 production cards)
and marked preliminary.

### C15. The page size is a format constant — detections snap to it

Architecture addition (Trond, 2026-09-08, after strips kept slipping
through as wrong-sized "pages"): the journal format guarantees uniform page
size, so size is never derived from a blob again. After all repair passes, a
final snap (`snap_pages`) turns every accepted detection into a full page
box: the blob gives position, `PAGE_SIZE_PRIOR` (2050×2780, measured across
13 production cards; pitch ~2180) gives the dimensions.

- **Prior resolution** (`resolve_page_size`): detections matching the prior
  ±10 % tune it (median); exactly one witness contributes its own clamped
  dimensions (using the raw prior was measured to shrink an edge-of-band
  card's pages) **and says so** — one witness out of many detections is a
  thin basis for a whole card; zero witnesses → per-card estimate with a
  LOUD note **naming the numbers** (how many detections, how many matched,
  what the card actually measures and how far off it is), so an off-format
  card is never silently forced into journal size. Card 029 in background
  mode built a whole card on 2040×2400, 14 % shorter than the format, and
  the log said only "not matched". The note now points upstream, which is
  where the cause was: two 10 px structure runs had cut its pages (C18). New format
  one day? Measure a healthy card's PAGE COORDINATES the same way and
  update the prior.
- **Cell assignment**: each detection belongs to the grid cell (row phase +
  pitch) nearest its center — not gap-chaining, because a right-hand strip
  of one page can sit closer to the neighbour page than to its own sibling
  (production card 612130000029). Strips anywhere INSIDE a cell are fine (a
  washed page may keep only its middle); a detection reaching into the
  neighbour PAGE's span is refused → exit 3. Phase comes from
  full-width members only; pitch is card-wide (one physical raster).
- **Deliberately NOT snapped**: single detections spanning >1.25 pages in
  either direction (unsplittable merges — snapping would shear content or
  invent a split; they keep their loud warning), everything on a card that
  is already failing (raw geometry preserved for diagnosis), and the header
  band (never part of the page grid).
- Snap growth is logged per page, marks the page blue, and does NOT count
  toward the over-repair limit — normalizing to the known size is normal
  operation; content verification is the planned occupancy check.
- **Row slots from the stripes** (steg 2, 2026-09-08, Trond's
  architecture): the dark edge-to-edge stripes between page rows that
  `remove_structure_rows` deletes are the row boundaries, and they are now
  returned and handed to `snap_pages` (full-res, also printed as
  `Structure rows (full-res y): …` in every report). A row's pages must lie
  in the slot between the stripe above and the stripe below. An anchor-less
  row (no full-height member) is anchored on whichever surviving edge keeps
  its box inside the slot — bottoms first (card 036: washed tops), tops when
  only that fits (card 111: only the top ~700 px of row 2 survived, and
  unconditional bottom-anchoring stacked the whole row on row 1 with
  align 100 / quality 90 — exit 3 in every report since RAPPORT-5).
- **Invariants, refused loudly:** no two page boxes may overlap, and no page
  box may cross a stripe; a slot shorter than a page is not a page row.
  Each violation is a `REFUSED …` note and the card exits 3 — a wrong guess
  upstream must never ship as a quiet page list.

### C16. Coverage guard — uncovered foreground caps the score

Field card 612130000036 scored **100.0** with its entire first page row
outside every box (see C17's row-building history). Now the share of
foreground mass outside all page boxes is measured on every run, in every
mode: above `COVERAGE_WARN_SHARE` (15 %) it prints a loud warning, caps the
quality score at `100 × (1 − share)`, and puts `COVERAGE: …` in both
visualization banners. It is the one signal that survives any upstream
mistake — never remove it to "clean up" the banner.

### C17. Background-first binarization (`--background-first`, flagged)

Trond's Photoshop principle: the jacket is the only *stable* class — content
varies wildly (faded, washed, half-dark) — so select the background and take
the complement. Foreground = |pixel − local jacket level| beyond
`BG_BAND_RATIO` (0.22) of the level, in **either** direction; the level is
the per-card p90 illumination field. Split/refine use the same rule via
local scalars. Polarity does not exist in this mode.

Mechanism choice (over direct blank-card diffing): jackets vary physically
card to card (brown/gray stripes), so a per-card statistical band is the
production mechanism and the **blank fasit card is the calibration**: its
jacket, texture and all, stays within ~0.25 of the local level; real content
sits at 0.35+; the faded-page fasit (28 % darker than jacket, invisible to
the global Otsu) is caught from 0.18 up. Hence 0.22.

Known limitations, why it stays FLAGGED until A/B-validated in production
(`RAPPORT.command` passes `--background-first` through to the segmenter):
it assumes the light-jacket journal type (on a Yamaha-type card the p90
field IS the pages), and the anonymized tape on the test fasit sits too
close to jacket level to detect — the tape was never representative of real
pages.

Field regression (all 2026-09-08 reports, 45 cards) is a committed test:
no refusals on passing cards, quality up across the board, worst card
612130000135 from 20.5 to 73.8.

Position witnesses (C15 addendum): sub-min-size blobs never build rows and
never vote on phase/pitch/edges, but one with real mass (≥ 0.5 % of a page —
a 20×10 speck of dirt once claimed a phantom cell on the real fasit) inside
an otherwise empty cell of an existing row claims a full page there (field
card 612130000098 lost half a row to the min-size filter).
Steg 3 (2026-09-08) tightened the witness rules on field evidence: the
witness page takes its **row's anchor y** (the band midpoint put 098's four
witness pages 1082 px below the rest of their row — quality 93.7 → 76.0
with four *correct* pages found); a witness must be page-like in **both**
dimensions (`WITNESS_MIN_DIM_SHARE`, 5 % of page width and height — real
field rests measure 520–1930 × 220–420, the 50×1990 sleeve edge that
claimed page 28 on card 135 does not); and the claimed cell must lie inside
the image and, on cards with ≥ 2 rows, inside the card's observed column
raster (135's phantom sat in a 13th column reaching past the image edge).
Ignored witnesses are logged (`position witness … ignored: <why>`).

The signature (`find_fragment_groups`): detections sharing an x-span
(interval IoU ≥ 0.8) with vertical gaps ≤ 15 % of the expected page height
are linked into transitive chains; within each chain (sorted by y) every
maximal contiguous window whose union height lands inside 0.8–1.8× the
expected height is a fragment group. Chains, not just pairs, because
production cards showed pages cut into stacks of 3–4 where no *pair* reaches
the union band (612130000111_00012); the windowing also keeps a tight
next-row neighbour from hiding a real stack by pushing the whole chain past
the band. Expected height is the tallest detection capped at 1.5× the
75th-percentile height — the tallest box is a whole page even when most
detections are fragments, and the cap keeps one unsplit vertical merge from
doubling the estimate. The union band is what separates a split page (union
≈ 1×) from whole pages in adjacent rows (union ≈ 2×, and row gaps also fail
the gap test). Both visualizations mark the suspect boxes in orange and the
banner says `SUSPECT FRAGMENTS`.

Known blind spot: a card whose *every* detection is a fragment of the same
kind has no whole page left to anchor the expected height. The generous union
band covers the measured fasit case (1.6×), but proportions beyond that
escape the guard.

### C22. Per-cell evidence is measured, and decides nothing (yet)

Every run logs one machine-readable line per cell of the card's raster,
**empty cells included**:

```
CELL row=1 x=2160 y=240 page=1 fg=0.929 edge=0.329
CELL row=1 x=2600 y=240 page=0 fg=0.000 edge=0.115
```

`fg` is the foreground share inside the cell measured on the **original
graytone**, not on the binary — the binary is exactly what failed on the
faded cards, so it cannot be its own witness. It follows the run's polarity
decision; without that it reads 0.000 on every real page of a Yamaha-type
card. `edge` is the share of pixels whose Sobel magnitude clears
`CELL_EDGE_LEVEL`.

**Why it exists.** A cell prior — laying out expected cells and looking for
evidence in each — needs a minimum evidence per cell, and the only per-cell
evidence logged until now was blob size, which cannot do the job: measured
over every witness rest in the field reports, correct cards run 0.8–9.3 % of
a page and card 203, which shipped **empty crops**, runs 0.5–8.5 %. They
overlap almost completely. So this contract measures and logs; it decides
nothing. The calibration comes from the field, where the empty cells of
short rows are the control that analysis never had.

**The empty cells must be empty CELLS, not the card margin** (steg 9F). The
first version enumerated on to the image edge, and healthy card
612130000531_00012 then reported its five `page=0` cells at x = 50–255 —
left of its first page at 2200, in the jacket and frame, where foreground
measures 0.39–0.48 and means nothing. Only the raster a row itself spans is
enumerated now; a cell the row *skips over* is the control we want. Extra
cells are opt-in (`margin_cells`), and the count left out is logged.

Nothing may act on these numbers until they have been calibrated against a
production run. Building a prior on an uncalibrated floor would be card
203's mistake with a better explanation.

### C23. Header content is not a page row

After step two the card's top band is often gone from the structure list —
three of the four affected field cards had only the *bottom* band left — and
the header text that sits **below** the 8 % mask line (C11) then stands as
detections and grows the card a sixth row.

We know where the header is, so `header_zone_detections` uses it. A
detection is header content only if **all** of these hold:

- it **starts within `HEADER_ZONE_REACH`** (0.25 page heights) of the mask
  line. Measured: the eight field blobs start 0–250 px below the mask, while
  the first real page row starts 1690–2670 px below it. This is what keeps a
  *short* first row from being eaten — a header blob continues the masked
  band, a page row starts a row gap below it.
- its **centre lies above** the topmost row carrying **at least two**
  full-height boxes. One tall blob is not an anchor; it could be the mistake
  itself. No such row at all: no anchor, nothing dropped, and the log says
  so rather than guessing.
- it matches the page prior in **neither** dimension. Width alone saves it:
  a first row crossing the mask keeps its page *width* while the mask cuts
  its top. Height alone saves it too: a fused row is several pages wide but
  page-*high*, and dropping it would lose four pages silently where the
  impossible-geometry guard (C12) fails the card loudly. The eight field
  blobs run 1.3–4.8 page widths at 0.46–0.81 page heights and match neither.

It runs inside `repair_and_snap`, so a report replays through it, and
**before** anything counts detections, so the evidence guard (C19) judges
pages against pages. Every dropped detection is logged with its size and
position; nothing is dropped silently.

### C21. The staircase: one diagnosed second threshold

C9 allows re-thresholding only as step two, after a measured trigger. This
is that step.

**Trigger** — validated on all 88 production cards:
`(0 detections OR the evidence guard refused the card) AND border share >
STEP2_BORDER_TRIGGER (40 %)`. Border share alone does **not** work and the
first proposal to use it would have refused correct cards: it is a fraction
of *total* foreground, so a card with few pages reads high even when
thresholding is perfect — card 418 has 5 pages, 85.5 % border and quality
100.0, and card 630 (which must pass) reads 76.4 %. Measured over the batch,
the composite trigger fires on exactly the 11 threshold failures and on no
card that passes today.

**Step two** (`otsu_excluding`) recomputes Otsu with the frame, the header
band and the border-connected structure taken out of the histogram — by
**position, never by level**, because the frame's colour varies from jacket
to jacket. On the synthetic replica of the field failure the global
threshold lands at 2 (frame against everything) and finds no page; masked,
it lands at 195 and finds every one. On a healthy card both give 33 — step
two is a no-op if it ever runs.

**It proves itself or the card is refused** — and the proof is that the card
**passes every guard on its own**, not a number about the frame. The *whole*
pass runs again from the new threshold (same code, no exemptions), and the
page size must come from the format prior rather than a per-card estimate.

The border share is logged before and after, and decides nothing. It used to
have to at least halve, which punished cards for having few pages: border is
a fraction of *total* foreground, so a small card reads high however well
the threshold worked. Card 612130000609_00036 was refused at 100 % → 61 %
while shipping 17 clean pages in 12+5 at quality 94.2 (steg 9C).

**Consequence worth knowing:** step two can only prove itself on a card in
the journal format, since the proof requires the page-size prior. A faded
card in a *different* format is refused even if step two recovered its pages
perfectly. That is deliberate — we cannot tell a recovered off-format card
from a mis-thresholded one — but it is the first thing to look at if a
genuinely different format appears in the batch.

**Never silent:** a `Step 2 threshold: trigger …, border … → …, otsu … → …`
line in the log, and `TRINN2` on the card's row in `SAMMENDRAG`. When step
two runs and still finds nothing, the card is told the reason it *had* —
`threshold found only the frame; re-threshold failed` — never "no pages
detected", which would blame the card for a threshold's mistake.

### C19. A card must be FOUND, not composed

Full production run of `65223c2`, 88 cards, 2026-09-08. A new failure class
shipped as `OK`: detection finds almost nothing, the witness and raster
machinery lays out a whole card from the rests, and the quality score
rewards it. Card 612130000203_00012 shipped **41 empty crops built from 9
detections at quality 84.8 GOOD**; 623_00012 shipped 2 pages from 1
detection at 100.0.

Counting witnesses, snap growth and geometry repairs as "invented" does
**not** separate those from healthy cards — measured across all 88, that
metric reads 95 % for 203 but also 85 % for card 135 and 94 % for card 630,
both of which are correct. Snap growth is normal operation (C15): those
pages exist, they are only normalised to the format size.

What separates cleanly is **how many pages come out per detection that went
in**:

| card | detections → pages | ratio | |
|---|---|---|---|
| 203_00012 | 9 → 41 | 4.6 | refused |
| 203_00024, 623_00012 | 1 → 2 | 2.0 | refused |
| 494_00024 | 4 → 8 | 2.0 | refused |
| 135_00012 | 23 → 27 | 1.17 | passes |
| 425_00012 | 33 → 40 | 1.21 | passes |
| 630_00012 | 28 → 35 | 1.25 | passes |

`EVIDENCE_MAX_PAGES_PER_DETECTION` is 1.5, in the gap. Above it the card
exits 3 with both numbers in the message. The ratio is logged on **every**
card, passing ones included, so the next report gives the distribution
rather than only the outliers.

`EVIDENCE_MIN_DETECTIONS` (3) is a floor for the case where the ratio
happens to stay low. It bites only when something *was* invented (pages out
> detections in): the real journal fasit card in `testdata/` has 2
detections and 2 pages and composes nothing.

### C20. The layout invariants refuse, they do not warn

A physical card holds at most `MAX_ROWS` (5) rows of at most
`MAX_PAGES_PER_ROW` (13) pages — a fasit from Trond, measured on the real
cards, not a heuristic. Exceeding either was a printed *warning* until
612130000432_00024 shipped **60 pages in six rows at quality 71.1 GOOD** in
the 88-card production run. A warning nobody reads is not a guard: the card
now exits 3 with the count in the message.

Judged on the repaired geometry (steg 5B), inside `repair_and_snap` with
the other geometry decisions, so a report can be replayed through it.
Measured across all 88 field cards: one new refusal in the archive
(432_00024, six rows) and four cards that already failed. Cards 517_00012
(12×5) and 227_00024 (12+13+11) pass — thirteen is the limit, not over it.

### C18. Full width is not enough — a stripe must prove itself

`remove_structure_rows` used to delete every full-width run thinner than a
page. That ate pages. A row of 12 inverted pages covers
12 × 2050 / 29071 = **84.6 %** of the width and `STRIPE_COVERAGE` is 0.85 —
the knife edge. Wherever the coverage dipped inside a row (a light band,
washed text) the row broke into runs each shorter than a page, and every one
of them was deleted as "structure".

Measured in the real A/B run (16 cards × both modes, 2026-09-08, code
`7eda4ed`): card 111 standard deleted 7470–9450 — *exactly* the missing
bottom of row 2, the geometry that steg 2's slot refusal had flagged. The
same mechanism hit 036 (3310–4400 inside row 1), 050 (6440–6870), 098
(6670–9240 and 10580–12660), 104 (3280–3540, 5700–5980) and 029 in
background mode (10 px runs at 4820 and 5520, which also dragged the card
onto a per-card page size of 2040×2400).

**The jacket is constant.** All 32 card runs carry the same seven structure
runs — a top band, five stripes, a bottom band — regardless of how many
pages the card holds or which mode produced it. So a run is structure only
if it is the top band (starts at or above the header mask), the bottom band
(reaches the image edge), or passes all three stripe tests:

| test | value | what it is measured against |
|---|---|---|
| thickness | ≥ `STRIPE_MIN_H` (80 px full-res) | real stripes are 100–420 px; the false slivers 10–20 px (029, 111, 050) |
| solidity | ≥ `STRIPE_SOLID_COVERAGE` (0.90) | field stripes measure 0.92–1.00, false runs 0.85–0.89 |
| position | within `STRIPE_RASTER_TOL` (150 px) of the card's stripe raster | the thick false runs (104's 260 and 280 px, 050's 340 px) sit 400–600 px off |

Thickness alone convicts nothing — **position is what convicts**. The raster
is fitted **per card** (`fit_stripe_raster`), never assumed: the first stripe
measures 5820–6370 across the field cards, the pitch 3360–3470. It is scored
on how **completely** it is filled and only then on how many candidates it
explains — a dense cluster of false runs inside one row (098 had eight)
otherwise supports a finer pitch that hits more candidates while leaving most
of its own positions empty.

The pitch is bounded above by `STRIPE_MAX_PITCH` (1.6 × the page-height
prior): a stripe raster is one page row plus a stripe, measured at 1.21–1.25
page heights. Without the bound, a card with **one missing stripe** hands the
fit to a raster of every *other* stripe — it fills perfectly (3 of 3) and
beats the real one (4 of 5), leaving the stripes between it in the binary and
dissolving a row slot. With fewer than three candidates no raster is fitted;
all of them are then kept as structure and the log says so.

Runs are coalesced across `STRIPE_MERGE_GAP` (60 px full-res) first: card 074
carries one stripe cut in two 40 px apart, while real stripes sit ~3400 px
apart. **Coverage is measured over the full-width rows only, never over the
bridged gap** — averaging the cut in reads 0.91 on 074's stripe and refuses
a real stripe as "not solid enough". A raster position with no candidate is
logged as a MISSING stripe — its slot then spans two rows, and the C15
invariants still hold.

Every run is logged either way, thickness and coverage included — and the
first A/B that carried those numbers moved the solidity floor from 0.95
(calibrated on the fasit cards alone, where stripes measure 0.99–1.00) to
**0.90**: in the field, 17 real stripes on cards 135 and 142 measure
0.92–0.95 at 100–160 px, while false runs never exceed 0.89. At 0.95 those
two cards lost every row slot they had, on cards that still passed.
Constants are defined in full-res px and stored as ratios of image height, so
they hold at detect scale too.

---

## Running it

```bash
.venv/bin/python -u segment_microfiche.py \
    -i /Users/m4-studio/NHA/Panoramas/612130000012_00016.tif \
    -O /Users/m4-studio/NHA/Microfiche/612130000012_00016
```

Use `python -u`. There is no explicit flushing, so piped stdout is block
buffered and a progress panel would otherwise receive everything in one lump at
the end.

### Flags

| flag | default | notes |
|---|---|---|
| `-i, --input` | — | panorama; anything libvips can open |
| `-O, --output` | `<input dir>/segmented/<stem>/` | the card folder |
| `-o, --order` | `rows` | `rows` = right then down (slik journaler leses); `columns` = down then right — matches no real card, kept for compatibility |
| `-hs, --header-skip` | `0.08` | fraction of height masked at top; this band becomes `page_000.tif` (C11). `0` disables both |
| `--no-header-page` | off | suppress the header band `pages/page_000.tif` (C11) |
| `--no-archive` | off | leave the panorama in place instead of archiving (C13) |
| `--archive-dir` | `../PanoramaArchive` | where an archived panorama goes |
| `-p, --padding` | `0.03` | crop margin; ≤1 = fraction of median page, >1 = pixels |
| `--refine` | off | re-detect each page locally at 20 %; slower, slightly looser |
| `--no-split` | off | keep merged detections instead of splitting at projection valleys |
| `--format` | `tif` | see C7 |
| `--invert` | auto | force inverted polarity (dark pages on light card) |
| `--no-invert` | auto | force normal polarity, disabling auto-detection |
| `--skip-extraction` | off | coordinates only; non-destructive (C4) |
| `--anon-viz` | off | also write `_debug/anon_viz.jpg`: solid black/white silhouettes of the detected blobs with boxes and banner, no readable content — safe to take off an air-gapped machine |
| `--debug` | off | write full-res binary TIFF to `<card>/_debug/` (box-overlayen `visualization.jpg` skrives alltid) |

`--anon-viz` exists because the production machine is air-gapped with journal
content that must not leave it: `visualization.jpg` and the binary TIFF show
readable text. Like `visualization.jpg`, the anon view is written on EVERY
run **including the failure paths** (exit 2/3) — failing cards are exactly
the ones that must be inspectable across the air gap; on failure it shows the
silhouettes of whatever survived detection plus a red `FAILED: <reason>`
banner. The anon view fills every detected blob solid (text inside a
page is a hole in the blob and gets painted over) while a gap that reaches the
blob's edge — a stitching seam splitting a page — survives, which is exactly
the diagnostic it exists to carry out. Overlay drawing is stamped in hard
colors; a test pins that the rendered image holds only black/white plus the
overlay palette, and another pins that no silhouette on the real card is
smaller than a page. Never add smoothing/closing to the mask or antialiased
drawing to the overlay.

### RAPPORT.command (m4-studio)

Double-click in Finder on the production machine: pick the folder holding the
panorama images, and every one is inspected (`--skip-extraction --anon-viz`,
touching neither sources nor existing card folders). Only whitelisted,
anonymized artifacts land in `~/Desktop/RAPPORT-<dato>/` — per-card text
logs, `<kort>_anon_viz.jpg`, and a `SAMMENDRAG.txt` with page counts, exit
codes, **the quality score and the detected grid per card**, and which cards
tripped the fragment guard. A card that exits 0 but grades POOR (below 60)
is listed as **`SVAK`**, never `OK`, and counted on its own line — card 111
read as plain `OK` at quality 52.5 while holding four pages in one box. `rapport.py` enforces the
whitelist on every copy and re-scans the finished folder; `visualization.jpg`,
binaries and page crops can never end up there. Cards that fail inspection
appear loudly as `FEIL` lines in the summary.

### TEST-RUNDE.command (m4-studio)

One orchestrated test round, from a list of card IDs to a comparison.
Double-click in Finder; the card list is `TEST-KORT.txt` beside the script,
one ID per line (`#` is a comment), or it asks for a file.

**The rule: only reports and anonymized artifacts leave the machine.** The
panorama copies are journal data, so they go to a **separate**
`TEST-PANORAMAER-<date>/` folder outside the report tree, with a
`LES-MEG-IKKE-KOPIER.txt` inside saying so. `TEST-RUNDE-<date>/` then holds
nothing but the two report folders and `SAMMENLIGNING.txt` — which makes
"copy the test round folder to the stick" a safe sentence. That separation
is for the *human*: the machine was already safe, since every copy goes
through the whitelist, but in the first version the panoramas sat beside the
reports and no whitelist helps against dragging the whole folder.

It **copies** each named panorama and never moves anything — a test round
must not disturb the production queue, so the sources stay exactly where the
app and the runner expect them.
Cards are looked for in `Panoramas/`, `Panoramas/error/`,
`PanoramaArchive/` and `Error/` under `sessionRoot` (read from
`~/.microfiche-station.json` like the app, default
`/Users/m4-studio/Desktop/NHA`), and the folder each was found in is
printed. Anything not found is listed loudly rather than silently skipped.

Then it inspects the copies in **both** modes into `rapport-standard/` and
`rapport-bakgrunn/`, and writes `SAMMENLIGNING.txt` against the previous
test round: one line per card with the outcome, page count, grid, quality
and `TRINN2`, marked `=` unchanged, `!` changed, `+` new, `-` gone — both
states on the changed line so a difference can be grepped, not just seen.
Per card it also carries the staircase (`TRINN2:` trigger, otsu and border
before/after, and whether step two proved itself) and the cell evidence
(`CELL` counts and median foreground for occupied and empty cells).

Only whitelisted, anonymized artifacts reach the stick, and each copy is
verified byte for byte. The whitelist is `rapport.py`'s, composed with the
one addition `SAMMENLIGNING.txt` — never duplicated, so it keeps a single
source of truth.

### VIS-PANORAMA.command (m4-studio)

The production panoramas are zstd-TIFFs that Preview cannot open.
Double-click, pick a TIFF (or Cancel to pick a folder), and a **lossless**
LZW viewing copy `<navn>_visning.tif` is written **beside** each source —
never overwriting, suffixed on collision — so real pixel values can be
measured with Digital Color Meter. Viewing copies contain journal data and
stay on the machine; they are never report artifacts.

### Replayable geometry in every report (2026-09-08)

Every run logs the coordinates the coordinate-only part of the chain works
on, so a report can be replayed without the panorama:

| line | what it is |
|---|---|
| `RAW detections (full-res x,y,w,h): …` | what detection found **before** the split pass — diagnosis only. Splitting reads the panorama, so no coordinate-only replay can reproduce it; the difference against the next line is exactly what splitting did |
| `PRE-REPAIR detections (full-res x,y,w,h): …` | the replay input: everything below it is pure geometry |
| `RAW witnesses (full-res x,y,w,h): …` | the sub-min-size blobs the snap uses as position witnesses (C15) |
| `Structure rows (full-res y): …` | the stripe runs, i.e. the row slots (C18) |

Feed those four through `repair_and_snap()` and the result is what the run
shipped. That function is the chain — `main()` only prints it and decides
the exit code — so a replay exercises production code, not a copy of it.
Coordinates only, no content, so the reports stay safe to carry off the
air-gapped machine.

With `--refine` the replay is not bit-exact (refine reads the image too);
production does not use it.

Every segmenter run starts with an `Env:` line (python/numpy/opencv/pyvips
versions, the code's short git SHA or `ukjent`, and the mode — `standard`
or `bakgrunn-foerst`) so each captured rapport.txt documents the environment
AND the code that produced it. `SAMMENDRAG.txt` carries the same `Kode:` /
`Modus:` header. Added 2026-09-08 after two same-day report sets differed
with nothing in either saying which code ran — and after the planned A/B
turned out never to have happened: `RAPPORT.command` forwarded only the
folder, so a flagged run silently became a standard run. It now forwards
every extra argument, and **`RAPPORT-BAKGRUNN.command`** is the
double-click B side (calls `RAPPORT.command` with `--background-first`).

## How detection works

1. Illumination-flattened Otsu threshold (2026-09-08, after mottled
   production panoramas silently lost pages to a global threshold): a
   ~600 px planning thumbnail yields a low-frequency illumination field
   (per-cell 90th percentile tracks the bright class — jacket or Yamaha
   pages — so page content does not read as lighting; floored so the dark
   surround cannot boost into fake foreground). Otsu is computed on the
   flattened thumbnail and applied to the full image as a threshold
   *surface* (`otsu × field / norm`), preserving the full-res
   threshold-then-resize order the detect pass depends on. The split and
   refine passes use the same field via local scalar thresholds. Flattening
   a flat image is ~identity; the share of thumbnail pixels it re-classifies
   is the mottle detector, **printed on every run**. Above
   `ILLUM_WARN_SHARE` the run prints a loud warning and both visualization
   banners say `UNEVEN ILLUMINATION`. That threshold was 0.5 % (from the
   fasit, where clean cards measure ~0.2 %) and is **3 % since 2026-09-08**:
   the field range for real production panoramas is 0.8–1.9 %, so 0.5 %
   warned on 100 % of cards — noise, not signal — while the one genuinely
   blotched card measured 6.5 %. Detection compensates either way; the
   warning tells the operator the *source* is sick.
2. Downscale to 10 %. Polarity is **auto-detected** per card (2026-09-04,
   decided cross-repo): the two known card types are opposite (Yamaha-type
   bright-on-dark, journal jackets dark-on-light) and the app sends no flag.
   Both polarities are tried through the cheap detect pass and scored on
   page-likeness — impossible spans (a box over `PAGE_MAX_SPAN` = 60 % of the
   image in either dimension) count *against* the polarity that produced
   them, since the wrong polarity reads full rows as pages. Border sampling
   cannot decide this: the dark mounting surround frames both card types, so
   the border ring reads background either way (measured). Auto-inversion is
   announced loudly in the log and in the viz banner; `--invert` /
   `--no-invert` force it.
3. **Remove card structure.** Full-width row-runs that are *proven* to be
   card structure are deleted (`remove_structure_rows` →
   `classify_structure_runs`, C18): the top band, the bottom band, and runs
   that are thick enough, solid enough and sitting on the card's stripe
   raster. Then erosion, then any
   remaining foreground *connected to the image border* is removed
   (`clear_border_connected`): the frame always reaches the border, pages
   never do. Validated against the real blank jacket, which must detect as
   exactly nothing. Connectivity alone is not enough — a sleeve can overlap
   its stripe (seen on the real card), which would drag the page into the
   border component; height/geometry is what separates structure from pages.
4. Contours → bounding boxes, filtered by minimum page size.
5. **Expand boxes by the erosion radius** (C1).
6. **Reject detections that are not page-shaped** (C12).
7. Scale back to full resolution and **split merged detections** — only
   boxes that could actually hold two pages (`can_hold_two_pages`: over 1.5
   pages in a dimension, since two pages side by side span ~2.1 page
   widths). The basis is the format page size, never the median of the box
   list, because after a split that list is dominated by the fragments
   themselves: card 098 split sixteen *single* pages into 2–4 fragments each
   and reported 38 normal pages as `~2 fused`. Touching
   pages fuse into one box at detect scale, but the gap between real pages is
   a projection *valley* — columns/rows whose foreground share drops below
   `SPLIT_VALLEY_RATIO` (0.6×) of the box's median. Each box is re-scanned
   against the source at 20 % — thresholding **after** the resize, which is
   what makes a dirty gap (the real card's overlapping tapes: 45 % foreground)
   read as background where the full-res-thresholded detect pass read it as
   page. Cuts that would leave an impossibly small piece are edge artifacts
   and are judged individually, so they never veto a real gap. Verified on
   the real card: the tape pair splits at the measured valley. `--no-split`
   disables. Then sort into reading order.
8. Optionally refine each box by re-detecting locally at 20 %.
9. Score the card (size consistency, **row** alignment, spacing, shape).
   The row and page-count warnings are judged on the **repaired** geometry:
   they used to count raw detections and fired on 10 of 16 field cards,
   every one of which shipped 12 pages or fewer per row.
   Rows are the only real axis — real cards hold at most 5 rows of up to 11
   pages, rows are NOT vertically aligned with each other, and there is no
   column structure (Trond, 2026-09-04). Nothing in the score rewards or
   punishes column geometry, and the reported layout is honest: `3 rows:
   4+4+4`, not an invented `NxM` grid. More than 5 rows or more than 13 pages
   in a row is warned about as likely misdetection.
10. Crop each page with the margin and write TIFFs in parallel (5 workers).
11. Write the header band as `page_000.tif` (C11).
12. Write `_done` (C2).
13. Move the panorama to `PanoramaArchive/` (C13).

## Tests

```bash
.venv/bin/python -m pytest test_segment.py -q
```

Unit tests cover the geometry, the band filter and the folder lifecycle.
The end-to-end tests drive the real CLI against a small generated card, so the
folder contract is verified in seconds without needing a gigapixel scan.

See `HANDOFF.md` for measured numbers, what is still unproven, and how to
verify a change with no network access.
