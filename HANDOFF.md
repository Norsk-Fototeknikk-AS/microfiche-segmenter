# Handoff — state as of 2026-08-22

Written before the machine goes offline for production. Read `README.md` first;
its "Contracts" section is the part that must not be broken. This file records
what was measured, what is still unproven, and how to check your work with no
network and no access to the conversations that produced this code.

Everything below was measured on this machine, not estimated. Dates are given
where they matter.

---

## Where the work happens

As of 2026-08-22 the live workflow runs on the **system disk**, not NB02:

```
/Users/m4-studio/NHA/Panoramas/     panoramas arrive here (our input)
/Users/m4-studio/NHA/Microfiche/    card folders are written here (our output)
/Users/m4-studio/NHA/Output/        Capture One's TIFF export
/Users/m4-studio/NHA/Error/
/Volumes/NB02/NHA/                  archive; RAW stays there, finished journals move there
```

`segmenterWatchRoot` in `~/.ocr-pipeline-config.json` points at
`/Users/m4-studio/NHA/Microfiche`.

Nothing in this repo stores a path — input and output both come from the command
line, verified by grep, so the move needed no code change. Paths in this document
are examples and historical records, not configuration.

The move was for **isolation, not speed**. Measured bandwidth differs by only
2.6-2.9x (NB02 2332 MB/s write, 1771 MB/s random read; internal 6669 and
2749), but *concurrent* use cost 16x — see "The 16x slowdown was not real".
Random read is what this tool does when pulling 147 crops out of a gigapixel
panorama, and NB02 was nearly as good at it. The disk was never the problem;
the neighbours were.

**Capacity, resolved:** the system disk has ~480 GB free and each panorama is
~1.14 GB, so archiving locally forever would fill it after ~420 cards. It does
not accumulate: panoramas are deleted at *journal end*, by the mover that copies
a finished journal to NB02. Only journals in flight hold one — roughly 24 at a
time, so about 55 GB. C13's default (`PanoramaArchive/` beside the input) is
correct as it stands.

Deliberately **not** deleted at segmentation. The panorama was needed again twice
on 2026-08-22 — once when a light table band became a false page 6, once when the
size filter went in — and both times `_done` was present and the card looked
finished. The quality score was the only hint, and it is information, not a gate.
PTGui deletes its tiles, so the way back from a lost panorama is re-stitching
from RAW, not a copy.

RAW is **not** currently written to `/Volumes/NB02/NHA/Capture` at capture time.
On 2026-08-22 that folder held 32 `.iiq` (4.1 GB), but their mtimes were 07:41
while their ctimes were 10:26 — they were shot elsewhere and moved in by hand
hours later. Until Capture One writes there during capture, RAW safety depends on
someone remembering to move files. It also covers the current round only; earlier
shoots went to the Capture One session Trash. Do not treat RAW as a general
safety net for anything already segmented.

(That mtime/ctime distinction cost a wrong conclusion here: a move preserves
mtime, so mtime answers "when was this shot", not "when did this arrive". Reading
one as the other is the same failure as the two-variable benchmark above — a true
number answering a different question.)

---

## What changed 2026-08-21

### The erosion bug (the important one)

Detection eroded the binary image to separate touching pages and never grew the
boxes back. Every page box sat **60 px inside the real page on all four sides**,
in the crops *and* in `page_coordinates.csv`.

Measured against the source with scanlines, before and after:

| edge | before | after |
|---|---|---|
| left | 50 px inside the page | 10 px outside |
| right | 51 px inside | 9 px outside |
| bottom | 46 px inside | 14 px outside |

Ground truth for page 1 of the test card: the page really spans x = 1010…2961.
The old box was 1060…2910. The new box is 1000…2970.

**Consequence for stored data:** coordinates written before 2026-08-21 are ~60 px
per side smaller than coordinates written after. Cards segmented before today are
not comparable with cards segmented after. If anything downstream cached
coordinates, they must be re-read, not diffed.

The card quality score also moved slightly (86.4 → 87.2 on the test card), because
grid alignment is computed from boxes that now match real page edges.

### Card folder lifecycle

`_done` was never deleted on a re-run — measured staying visible for the entire
10 s rewrite, which would hand the OCR app a half-written card. Stale page files
were never purged. Both fixed; see contracts C2 and C3.

### Debug artifacts moved

`temp_binary.tif` (~9 MB) and `visualization.jpg` used to be written into the
card folder, where the app's fallback could read them as pages. Now `--debug`
only, into `<card>/_debug/`.

### Fail loud on zero pages

Was: empty folder + `_done` + exit 0, which the app ignores silently — a failed
card vanished with no error anywhere. Now: no `_done`, source moved to
`<input dir>/error/`, empty card folder removed, exit 2.

### Two latent crashes found and fixed

- `compute_card_quality()` returned a dict **without** the `'grid'` key when
  fewer than 2 boxes were found, while `main()` printed `quality['grid']`
  unconditionally. Any card yielding 0 or 1 pages died with `KeyError` mid-print,
  before the error handling could run. Not reachable on a normal card, very
  reachable on a bad one.
- The startup cleanup, as first written, also ran under `--skip-extraction` —
  so inspecting a finished card **deleted all of its pages**. Caught by an
  end-to-end test, not by inspection.

---

## What changed 2026-08-22

### The header is no longer thrown away

`--header-skip` masks the top of the card so the header is not detected as a
page. It was then discarded entirely — the card's only identifying text, gone on
every run. It can now be written as `pages/page_000.tif` at 1/16 scale via
`--header-page`. See contract C11 in the README for why the name, the scale, and
the boundary at `header.json` are all fixed points rather than preferences.

**On by default since 2026-08-23.** It shipped off for one day while the import
side was built, then was switched on once that side proved it handles page zero:
294 journal pages counted from 296 files across two cards, with both headers
carried as sidecars. Downstream deliberately supports both shapes, because a
half-finished handover can contain cards from either side of the change, and that
kind of transition is what breaks quietly.

`--no-header-page` suppresses it. `--header-page` survives as a no-op: argparse
exits 2 on an unknown flag, and 2 is EXIT_NO_PAGES, so a caller passing the old
flag would look exactly like a card that detected nothing.

What the header is for, decided the same day: it is OCR'd with a field-oriented
prompt (a card header is fields, not prose), the text is stored per card, and the
image is then carried as a sidecar rather than deleted — on a name or national ID
conflict with the pages, **the header wins**, and the discrepancy is flagged for
review with the header image shown next to the disputed field.

### Segmented panoramas leave the work queue

Contract C13: on success the panorama moves to `PanoramaArchive/` beside
`Panoramas/`, after `_done`. Watch the re-run consequence — the input is no
longer where it was, so re-running a card means pointing at the archived copy or
having passed `--no-archive`. Two existing tests had to opt out of archiving for
exactly this reason, which is a fair warning about how it changes habits.

### Non-page-shaped detections are rejected

`drop_band_detections()` (then `drop_size_outliers()`), contract C12. Added after both production cards came out
as 148 pages / POOR because of a light table band along the bottom edge.

## Proven since first writing

### Multi-card journals work (2026-08-22)

Two cards of one journal, same fanearkID, distinguished only by the suffix:
`612130000012_00016` and `612130000012_00032`. This is the case where a bare
`<fid>` name would have made card 2 delete card 1's pages in the startup
cleanup.

Measured, not assumed: card 1's folder was snapshotted (151 files, sizes,
mtimes, inodes) before card 2 ran, and compared after. Nothing missing, nothing
added, nothing modified, and `_done` kept the same inode — it was not even
rewritten. The only diff was `.DS_Store`, which Finder touches and we never
write.

The startup purge was also confirmed in production the same day: a stale
`page_148.tif` left by an earlier run was gone after the next run.

### The 16x slowdown was not real

Two runs took 2 min 10 s where the same work had taken 8.2 s, and it was briefly
blamed on the input TIFF's layout. That was wrong twice over — the files are
identical in encoding (`AdobeDeflate`, Rows/Strip 128, striped) and pure decode
cost matches to within 0.01 s. The full matrix:

| input | output | wall | user CPU |
|---|---|---|---|
| yesterday's file | NB02 | 8.2 s | 29 s |
| today's file | NB02 | 2 m 13 | 547 s |
| yesterday's file | local disk | 8.18 s | 29.3 s |
| today's file | local disk | 8.16 s | 29.1 s |

Only the combination was slow, and re-running it later gave 8.15 s — it does not
reproduce. The slow runs coincided with PTGui stitching and compressing on the
same volume (217 s per panorama). Contention, not code.

**Real cost is ~8 s per card.** Do not build timeouts or estimates on the
2-minute figure. Beware of "isolating" a variable while changing two: the first
attempt here swapped the input file *and* the output destination, and produced a
confident wrong conclusion.

## Measured numbers

### Production card `612130000012_00016`

```
input    Panoramas/612130000012_00016.tif   (on NB02, before the move)
         34354 × 25533 (877 MP), 1.14 GB, 3 bands
output   Microfiche/612130000012_00016/
         147 pages, 730 MB, grid 16×13
quality  88.1/100 GOOD  (size 98.2, align 76.1, spacing 96.6, shape 88.9)
runtime  8.2 s wall, ~380 % CPU
otsu     111
```

Crop accuracy, every page measured against its true edge by scanline:
**zero pages clipped on any side.** Worst case sits 2 px *outside* the page
(left 7 px, top 13 px) — all paper retained, hairline of card background.

### Test card (Desktop, Yamaha RD500LC manual)

34354 × 25548, 147 pages, 16×13, quality 87.2, otsu 112, ~10 s. Same physical
card as the production one, shot earlier under the wrong fanearkID.

### Cards segmented 2026-08-22 (with the size filter)

```
612130000012_00016   147 pages   Card Quality 91.8/100 GOOD   grid 16x13
612130000012_00032   147 pages   Card Quality 92.7/100 GOOD   grid 16x13
```

Both initially came out as 148 pages / POOR (58.8 and 59.7) because of a light
table band along the bottom edge — 33190 x 720 and 33180 x 720. With
the size filter (now `drop_band_detections()`) both land at 147 pages and GOOD. Size consistency went
from 0.0 to 98.5 / 98.6.

Both predate the header prepage, so neither has a `page_000.tif`.

### Constants that matter

| thing | value | full-res equivalent |
|---|---|---|
| detect scale | 0.1 | — |
| detect erosion | 7×7, 2 iterations | 6 px → **60 px** per side |
| refine local scale | 0.2 | — |
| refine erosion | 3×3, 2 iterations | 2 px → **10 px** per side |
| default margin | 3 % of median page (1 % until 2026-09-03) | 61 × 49 px on the production card |
| band ratio | 2.5× median in one dimension, ≤ 0.75× median in the other (sliver test since 2026-09-03; was a 40 % size tolerance) | rejects stitch edge bands, keeps merged rows (C12) |
| header scale | 1/16 | 34298 × 2038 band → 2144 × 127, 424 kB |
| extraction workers | 5 | — |

The header scale is pinned by a downstream constraint, not by taste: the OCR
pipeline's packaging step caps page width at 2480 px, so 2144 px passes through
untouched while anything larger is scaled back down immediately. If header OCR
of small print turns out unreliable, the fix is likely to make the header an
exception to that cap rather than to raise this number in isolation.

`--refine` on the production-grade card reports avg shift 6 px x / 11 px y from
the global pass, i.e. the two passes now agree. Before the erosion fix they
disagreed by ~50 px. Refine is slightly *looser* than the default path
(−49 px vs −10 px on the left edge of page 1) — both safe, default is tighter.

### The resize/threshold order duality

The same gap gives two different answers depending on operation order, and the
split pass exists in that difference (2026-09-04). The detect pass thresholds
at FULL resolution, then downsamples the 0/255 map: a dirty gap (the real
card's overlapping tapes, 45 % dark) classifies pixel-by-pixel as mostly page,
so the neighbors merge. The split scan downsamples the gray FIRST, then
thresholds: the same gap averages to a light value and reads as background —
the valley. Neither order is "correct"; the detect order is robust for finding
pages, the scan order is sensitive to gaps. Changing either order breaks the
mechanism it serves.

### Fragment guard, exit 3 (2026-09-07)

~Half the production cards after the m4-studio upgrade got pages cut
horizontally in two detections (top ~1/3 + bottom ~2/3) by light stitching
seams — archived as SUCCESS with shifted numbering. `find_fragment_pairs`
detects the signature (x-IoU ≥ 0.8, gap ≤ 15 % of expected page height, union
0.8–1.8× expected) and the card exits 3 without `_done`, source to `error/`,
suspect boxes orange in both visualizations. Expected height = tallest box
capped at 1.5× the 75th-percentile height; see C14 in the README for why each
piece is shaped that way and for the all-fragments blind spot. Merging the
halves instead is deliberately NOT done — content may be missing in the seam
gap (possible phase 2 after real seamed cards are inspected with
`--anon-viz`). MicroficheStation must treat exit 3 as failure — agreed with
the coordinator session 2026-09-07.

### RAPPORT.command + rapport.py (2026-09-07)

Finder-first report extraction on m4-studio: inspect every panorama in a
chosen folder, collect ONLY whitelisted anonymized artifacts (per-card
rapport.txt, anon_viz.jpg, SAMMENDRAG.txt) into `~/Desktop/RAPPORT-<dato>/`
for the USB stick. Inspection runs into a TemporaryDirectory precisely
because that output contains the non-anonymized visualization - the report
folder only ever receives files through `copy_safe`, and is re-scanned at the
end as defense in depth. The .command wrapper derives the repo path from its
own location (`$0`) and calls `.venv/bin/python` absolutely - launchd/Finder
gives no usable PATH.

### Anon viz on the failure paths + rapport guard (2026-09-07, evening)

Field bug from m4-studio: error-path cards (exit 2) got text reports but no
anon_viz - main returned at the STEP-3 fail branch before the viz step ever
ran. Now the fail branch writes anon_viz too (silhouettes of whatever
survived detection + red FAILED banner), same every-run principle as
visualization.jpg. And rapport.py's SAMMENDRAG forces any card without an
anon_viz to a FEIL row with "(anon_viz mangler)" - a missing expected
artifact must never read as success, whatever the exit code said.

### Guard extended to chains + quality warning (2026-09-07, late)

Report analysis of 81+6 production cards confirmed the seam hypothesis
(scattered horizontal blending damage, one cluster per page row, varying per
card - re-stitching is the fix, auto-merge is shelved because content is
EATEN, not just displaced). It also caught a guard blind spot in the wild:
612130000111_00012 passed OK with pages split into stacks of 3-4 fragments -
no PAIR reaches the union band. `find_fragment_pairs` became
`find_fragment_groups`: transitive chains over the same x-IoU/gap criteria,
then maximal contiguous windows inside the union band (windowing keeps a
tight next-row neighbour from pushing a real stack past the band).
SAMMENDRAG additionally warns `ADVARSEL LAV KVALITET` on any card scoring
below 50 regardless of exit code - in the field run every sick card was
below 50, every healthy one above 74.

### Illumination-robust thresholding (2026-09-08)

Production panoramas came out MOTTLED after a machine upgrade (patchy
brightness, deterministic, content intact). The global Otsu threshold put
patches on the wrong side - swiss-cheese binaries; the chain guard fired
correctly on bad binaries, and the old path silently ATE pages (measured:
half of 36 gone on the synthetic fixture, 4 of 12 on the first probe).
Fix: illumination flattening. Field = per-cell p90 of a ~600px planning
thumbnail (tracks the bright class; content does not read as lighting),
blurred, floored at 0.4x max (dark surround must not boost). Otsu on the
flattened thumbnail, applied full-res as a threshold SURFACE otsu*field/norm
- the threshold-first-then-resize duality is preserved. Split/refine use
local scalar thresholds from the same field. Mottle detector = share of
thumbnail pixels flattening re-classifies (field ratio does NOT work - the
dark surround gives 1.9 even on clean cards): clean 0.15-0.19%, blotched
fasit 0.93%, warn at 0.5% -> loud stdout warning + UNEVEN ILLUMINATION in
both banners. Two things the estimator depends on: field cells (12 across)
must stay COARSER than a page - real pages are ~1/14 of card width - and
the planning thumb is a target WIDTH (600px), not a fixed scale, so
structure stays resolved on small images. Clean real panorama re-measured
after the change: same two pages, 0.2% re-classified, no warning; the tape
pair now separates already at detect (the knife-edge gap flips with any
epsilon threshold change - final result identical).

### VIS-PANORAMA.command (2026-09-08)

Lossless LZW viewing copies of the zstd-TIFF panoramas, written beside the
sources, never overwriting. On-machine only (journal data). pyvips in the
venv reads the production zstd files - proven by the segmenter reading them
- even though the LOCAL libvips here lacks zstd write support (why the test
uses a deflate source).

### Phase 2: geometric completion (2026-09-08, evening)

The guard's exit 3 flipped to a repair step for everything that reconciles
with the page grid - Trond's call, grounded in field data (28 production
groups: all x-IoU 1.0, gap 0, union ~2790-2800 = one page height, so every
one resolves by sheet size). `complete_geometry` merges grid-matching
chains (union box; refuse over 30% invented area - the group criteria
already bound invention near that, the cap is a backstop) and extends lone
short detections to their row's anchors (>=2 full neighbours; top-edge
anchored - pages share tops within a row). Repaired pages: blue in both
viz, counted in banner, logged with invented share. Guard re-runs AFTER
repair - refused merges re-detect and still exit 3, and repairing more
than 50% of a card exits 3 (field worst: 28%, card 612130000135 with 34
detections/7 groups - now a unit-test fixture). Refused fragments are also
excluded from extension: contested geometry must not be quietly repaired
by the other mechanism. Contract change: seamed-card e2e tests flipped
from exit-3 to completed-with-blue.

### Vertical stripe merging (2026-09-08, late)

Trond overrode the risk log: strips repair NOW. find_stripe_groups
transposes x<->y and reuses the whole chain machinery; only the union band
differs (0.8-1.2x expected page WIDTH - neighbours union ~2x plus a real
gap, and THAT exclusion is the entire risk; pinned by a field-pitch row
test). Runs on the horizontally-repaired boxes, so a quadrant-split page
heals fully. Field fixture: card 135 detections 11+12 (1350+720 wide, gap
0) merge to one 2070-wide page. Two half-width documents in neighbouring
frames never link - frame spacing exceeds the 15% gap criterion.

### Phase 3: sub-band chains (2026-09-08, night)

Field-confirmed (pair 17+27, union 0.79x): the lower union bound is gone
for MERGING (find_fragment_groups grew a union_min parameter; the guard
still uses 0.8) - a sub-band chain merges and its short union is completed
to the row's anchors by the extension pass (force flag on the entry; the
0.7 short-ratio does not apply to it, row height does). The strengthened
tests exposed why merging must come first: extension alone repaired the
top piece and left the sibling as an overlapping GHOST page. Refused
groups now fail the card EXPLICITLY in main - a refused sub-band chain
does not re-detect under the guard's 0.8 bound. Upper bound 1.8x stays
(cross-row); field row gaps (840-920px) are ~2x the gap criterion, pinned
by a row-boundary test with real coordinates.

### Page-size prior + grid snap (2026-09-08, architecture addition)

C15 in the README carries the model; what the file history adds: the snap
was ordered as an ADDITION (existing passes untouched, Trond: "mye av det
vi har gjort er riktig"), and it runs BEFORE the guard's re-check because
cell assignment reunites what the chain criteria cannot (a wash wider than
the 15% gap allowance splits a page into pieces the chains refuse to link,
and the extension pass alone leaves the sibling as a ghost page - found by
the washed-panorama e2e, exit 3 before the reorder). Cell assignment is by
detection CENTER, phase from full-width members only - gap-chaining
misassigns right-hand strips (card 612130000029: strip 11 glued to page 12
instead of its own page 10), and one off-grid slot used to poison the
phase median. A member may reach into the inter-page GAP (dirty seams put
split cuts mid-gap) but never into the neighbour page. Field regression
over both reports is a skipif-guarded committed test.

### Background-first round (2026-09-09): root causes and five components

Root causes from RAPPORT-2026-09-08-5, identified as ordered (not just
made to disappear):
- 036 = ROW-BANDING COLLAPSE in snap_pages, NOT the header skip: the washed
  first row's detections had low tops, so row 2 fell inside the page-height
  span band -> one band, every cell y-anchored on row 2, a whole page row
  uncovered - at quality 100.0. Fixed by transitive y-OVERLAP row
  clustering + bottom-anchoring for anchor-less rows + the C16 coverage
  guard (which alone would have exposed it).
- 111 = the same banding built a phantom second row from bottom fragments;
  overlap clustering folds them into their own row's cells.
- 098 = half a row of small rests died in the min-size filter before the
  snap could see them -> position witnesses (C15 addendum, with the
  0.5%-of-a-page mass floor measured against a real 20x10 dirt speck that
  claimed a phantom cell).
- 135's faded pages = clearly visible contrast, below the one-sided global
  threshold -> C17 background-first (flagged; validated against the blank
  fasit, a synthetic faded card and pasted-page substrates; A/B in
  production via RAPPORT.command --background-first).
Also: the over-repair limit now counts only SUBSTANTIAL repairs (>=3%
invented, minimum 3 of them) - split-then-remerge churn at 0% invented is
bookkeeping, and a single repair on a two-page card is not "most of the
card".

### Tests

172 tests, ~19 s (was 32 when this was written). Unit tests for box
geometry, folder lifecycle and band filtering; end-to-end tests drive the real
CLI against a generated 4×3 card.

### Anonymized diagnostics off the air-gapped machine (2026-09-07)

`--anon-viz` writes `_debug/anon_viz.jpg`: the detected blobs as SOLID
black/white silhouettes (external contours filled, dilated by the erosion
radius) with the usual boxes, numbers and quality banner stamped on top in
hard colors. Content inside a page is a hole in the blob and gets filled shut;
a gap reaching the blob edge — a stitching seam that split a page — is not a
hole and stays visible. That asymmetry IS the feature: geometry and seams go
out on the USB stick, journal content does not. Two safety pins in the tests:
the rendered image may hold only black/white plus the overlay palette (OpenCV
5 antialiases text regardless of lineType, hence the mono-layer stamping in
`_stamp_solid`), and no silhouette on the real card may be smaller than a
page. Do not add closing/smoothing to the mask or draw directly on the output.

---

## Risk log: card 135's row 1 sits on five different y values

Found 2026-09-08 while pinning the physical page count. In the shipped
production output of `65223c2` (RAPPORT-2026-09-08-8, standard mode), card
612130000135's first row holds pages at **y = 3040, 3050, 3130, 3210 and
3240** — a 200 px spread — and its third row at **9940 and 10170**. The
fasit says pages share their top edge within a row.

The page COUNT is verified correct: Trond counted 27 on the physical card
and the run produced 27. So this is not about how many pages, but about
where their crops sit. Two hypotheses, not yet separated:

1. **Real per-cell skew.** 135 is the faded card; each cell kept its own
   surviving edge, and that is the best evidence available per page.
2. **The anchor giving way to noise.** The row consensus lost to per-cell
   evidence that was itself unreliable.

**How to tell them apart, no code needed:** open the crops from 135's first
row in Finder after the batch run. If the tops of the pages are cut, the
anchor is wrong (hypothesis 2). If there is jacket above the text, the crops
are merely generous and the skew is real (hypothesis 1). That answer is the
intended basis for the still-unordered occupancy check. Nothing is built for
this yet, deliberately.

---

## What is UNPROVEN

Do not assume any of this works. None of it has been exercised.

- **A FULL journal card.** The sharpest open risk (2026-09-04). Structure-row
  removal (`STRIPE_COVERAGE` 0.85) is validated on a *sparse* card whose page
  rows sit at ~20 % coverage. On a full card, page rows can reach ~96 % —
  everything then rests on the height rule (page-height runs are kept) and on
  the stripe-to-row gaps actually dipping below the threshold between runs.
  **Operator step until proven: the first FULL journal card goes through
  `--skip-extraction` with a visual check of `_debug/visualization.jpg`
  BEFORE any ordinary run.**

- ~~**Multi-card journals.**~~ **PROVEN 2026-08-22** — see below.
- **Real archival material.** Partly addressed 2026-09-04: a real journal
  jacket (no patient info; two anonymized tape-pages at accurate geometry) is
  committed at detect scale in `testdata/` with pinning tests. It differs from
  the Yamaha card in every way that matters: light jacket, dark pages
  (requires `--invert`), edge-to-edge stripes between rows, sleeves that can
  overlap stripes. Still unproven: real page texture (tape is uniform; fiche
  negatives are not), full cards, and page-to-page contrast.
- ~~**`--order rows`.**~~ Confirmed correct 2026-09-04, from the domain:
  real cards are rows-only — at most 5 rows of up to 13 pages (12 measured
  in production 2026-09-08, +1 margin), no column
  structure, rows not vertically aligned. `--order columns` matches no real
  card and is kept only so existing callers do not break.
- ~~**`--invert`.**~~ Exercised on the first real journal card 2026-09-04 —
  and it was **broken** (uint8 wrap made everything foreground; fixed same
  day). The real journal cards are dark-pages-on-light-jacket. Resolved
  cross-repo the same day: polarity is **auto-detected** per card (the app
  stays flagless); `--invert` / `--no-invert` remain as manual overrides.
  Pinned against all three ground truths (journal, Yamaha-type synthetic,
  blank jacket).
- **`--format jpg`.** Not run since today's changes. Forbidden for pipeline
  output anyway (contract C7), but the code path exists.
- **Zero-page failure in production.** Verified with a synthetic specks-only
  card, never on a real bad scan. Note the failure path *moves the operator's
  source file* — that behaviour has only ever run against throwaway inputs.
- ~~**The `error/` folder location.**~~ **RESOLVED 2026-09-08** (coordinator).
  It was never a disagreement — they are two stages:
  - `NHA/Error/` (capital E) is the **stitch runner's** folder for failed
    TILE SETS, one folder per set with a `REASON.txt`. The app's `errorDir`
    points there.
  - `Panoramas/error/` (lowercase) is the **segmenter's** folder for failed
    PANORAMAS (C9, C13).
  A panorama therefore never lands in `NHA/Error/`. `test_runde.py` searches
  it anyway when locating a card — it costs nothing and makes the layout
  visible rather than reporting a card as missing.
- ~~**Output root.**~~ Confirmed 2026-08-22: `segmenterWatchRoot` is
  `/Users/m4-studio/NHA/Microfiche` in `~/.ocr-pipeline-config.json`. It was
  first chosen by inference as the NB02 equivalent, before that config file
  existed, and turned out right; it then moved with the rest of the workflow.
- **header.json.** We deliberately do not produce it, and that boundary is
  agreed: this repo produces the header *image*, the OCR pipeline interprets the
  text. Do not invent a schema for it.
- **Header OCR quality.** Nobody has yet OCR'd `page_000.tif`. The plan is to
  read name and national ID from it — printed far larger there than on the pages
  — store the text, drop the image, and let the header win on conflict with a
  flag for review. Whether 1/16 scale is enough for reliable ID digits is
  untested.
- **The "N of M" mismatch check.** The header's card index is the only known way
  to catch an operator forgetting to chain a second card, which otherwise files
  it under the wrong patient undetectably. Nothing checks it yet.

---

## Pitfalls

- **A green test suite does not prove the geometry is right.** The erosion bug
  passed every test that existed. Geometry changes must be verified by measuring
  crops against the source image (recipe below).
- **stdout is block buffered.** No explicit flushing anywhere. Run with
  `python -u` or `PYTHONUNBUFFERED=1` if anything parses progress output.
- **`-O` is the card folder, not the root.** Easy to get wrong; produces a
  correct-looking folder in the wrong place.
- **`pages/` holds one more file than the reported page count.** The count comes
  from detected boxes; the folder also holds the header as page zero. 147
  reported, 148 files. This is intentional — see C11.
- **The quality score is information, not a gate.** A card scoring 30/100 is
  processed exactly like one scoring 95/100. Only *zero pages* fails.
- **Detection never sees a barcode.** fanearkIDs come from the Capture One
  AppleScript chain (`~/Projects/CaptureOneNamer`, counter over `names.txt`),
  not from anything on the card. If IDs look wrong, the cause is upstream —
  no rule applied to filenames here can recover a correct ID.
- **Do not add fanearkID validation here.** This tool sees one finished
  panorama, never the 16 tiles it was stitched from, so it cannot tell a card
  whose tiles carried mixed or unissued IDs from a clean one — it just inherits
  the name verbatim (C6). The OCR pipeline's preflight runs before stitching,
  sees the tiles, and fails on both mixed prefixes and IDs absent from the
  register. That is the right place for it; duplicating it here would only be
  able to guess.

---

## Verifying a change offline

No network needed for any of this.

**1. Run the suite.**

```bash
.venv/bin/python -m pytest test_segment.py -q      # expect 32 passed
```

**2. Run against a real card and check the folder contract.**

```bash
.venv/bin/python -u segment_microfiche.py -i <panorama> -O /tmp/check
ls -a /tmp/check          # expect exactly: _done  page_coordinates.csv  pages
head -1 /tmp/check/page_coordinates.csv    # must contain "Card Quality: N/100"
```

**3. Measure crops against the source — the check that catches geometry bugs.**

For a page box `(x, y, w, h)` from the CSV, take a full-resolution scanline
through the page's vertical centre, threshold it at the Otsu value the run
printed, and find the bright run containing the box centre. That run is the
true page extent.

```
clip_left  = x - true_left          # > 0 means the crop cuts into the page
clip_right = true_right - (x + w)   # > 0 means the crop cuts into the page
```

Both must be **≤ 0** for every page. On the production card the worst value is
−2 px. Reject any change where a page goes positive. Match only runs narrower
than about 1.4 × the box, otherwise two pages merge across the gutter and the
measurement silently reports "safe".

**4. Re-run twice and confirm idempotence.**

```bash
touch /tmp/check/pages/page_999.tif
.venv/bin/python segment_microfiche.py -i <panorama> -O /tmp/check
ls /tmp/check/pages/page_999.tif   # must be gone
```

---

## Open items

- The two cards under `Microfiche/612130000029_*` were segmented **before** the
  header default flipped, so they have no `page_000`. Their journals were mid-
  pipeline at the time; re-segmenting a card mid-run purges `pages/` and would
  break it. If headers are wanted for them, do it after those journals finish and
  before the mover deletes their panoramas from `PanoramaArchive/`.
- The OCR-Pipeline app now runs segmentation itself, so this repo may not need to
  be invoked manually at all. Coordinate before writing into the watch root.
- The pre-fix output folder on the Desktop
  (`~/Desktop/NHA Mikrofiche/Output/segmented/612130000333_0000736115 Panorama/`)
  still holds coordinates from before the erosion fix, plus a `_done` that the
  OCR app may act on. It was left alone deliberately — unknown whether the app
  already consumed it.
- Where the archive should live is **undecided**: C13 defaults to
  `PanoramaArchive/` beside the input, which is now the system disk. NB02 is the
  designated archive volume and has 7.4 TB against the system disk's 480 GB.
- `main` may be ahead of `origin/main` and unpushed. Check before assuming the
  remote has this work.
