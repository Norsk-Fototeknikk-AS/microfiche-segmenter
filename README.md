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
No re-thresholding heuristics: no real panorama has failed this way yet, so
the failure is reported, not guessed around.

### C10. Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | unhandled exception (traceback on stderr) |
| 2 | no usable detection (zero pages, or degenerate threshold); source moved to `error/` |

`-O` / `--output` names the **card folder**, not the watch root. A caller that
wants `<root>/<fid>/` must pass `-O <root>/<fid>` itself.

### C11. The header band is kept as page zero

`--header-skip` masks the top of the card during detection so the header is not
found as a page. That band is written out as `pages/page_000.tif`.

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
flat. A merged row is kept and warned about (`suspected merged pages` in the
log); splitting it is a known follow-up, pending a real failing card to
verify against.

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
| `-o, --order` | `rows` | `rows` = right then down (slik journaler leses); `columns` = down then right |
| `-hs, --header-skip` | `0.08` | fraction of height masked at top; this band becomes `page_000.tif` (C11). `0` disables both |
| `--no-header-page` | off | suppress the header band `pages/page_000.tif` (C11) |
| `--no-archive` | off | leave the panorama in place instead of archiving (C13) |
| `--archive-dir` | `../PanoramaArchive` | where an archived panorama goes |
| `-p, --padding` | `0.03` | crop margin; ≤1 = fraction of median page, >1 = pixels |
| `--refine` | off | re-detect each page locally at 20 %; slower, slightly looser |
| `--format` | `tif` | see C7 |
| `--invert` | off | for cards that are dark-on-light |
| `--skip-extraction` | off | coordinates only; non-destructive (C4) |
| `--debug` | off | write full-res binary TIFF to `<card>/_debug/` (box-overlayen `visualization.jpg` skrives alltid) |

## How detection works

1. Otsu threshold from a 1 % thumbnail, applied to the full image.
2. Downscale to 10 %. With `--invert` (dark pages on a light jacket — the real
   journal card type, first seen 2026-09-04) the binary is inverted here.
3. **Remove card structure.** Full-width row-runs that are thinner than any
   possible page or touch the image boundary / header mask are stripes and
   edge bands — deleted (`remove_structure_rows`). Then erosion, then any
   remaining foreground *connected to the image border* is removed
   (`clear_border_connected`): the frame always reaches the border, pages
   never do. Validated against the real blank jacket, which must detect as
   exactly nothing. Connectivity alone is not enough — a sleeve can overlap
   its stripe (seen on the real card), which would drag the page into the
   border component; height/geometry is what separates structure from pages.
4. Contours → bounding boxes, filtered by minimum page size.
5. **Expand boxes by the erosion radius** (C1).
6. **Reject detections that are not page-shaped** (C12).
7. Sort into reading order, scale back to full resolution.
8. Optionally refine each box by re-detecting locally at 20 %.
9. Score the card (size consistency, grid alignment, spacing, shape).
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
