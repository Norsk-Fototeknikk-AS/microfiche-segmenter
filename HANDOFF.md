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

**The flag is off by default for now** — Trond deferred the header to the next
round so the import side can be finished and tested against a card that has no
`page_000` first. Downstream deliberately supports both shapes, because a
half-finished handover can contain cards from either side of the change, and
that kind of transition is what breaks quietly. Turn it on when the import side
is ready.

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

`drop_size_outliers()`, contract C12. Added after both production cards came out
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
`drop_size_outliers()` both land at 147 pages and GOOD. Size consistency went
from 0.0 to 98.5 / 98.6.

Both predate the header prepage, so neither has a `page_000.tif`.

### Constants that matter

| thing | value | full-res equivalent |
|---|---|---|
| detect scale | 0.1 | — |
| detect erosion | 7×7, 2 iterations | 6 px → **60 px** per side |
| refine local scale | 0.2 | — |
| refine erosion | 3×3, 2 iterations | 2 px → **10 px** per side |
| default margin | 1 % of median page | 20 × 16 px on the production card |
| size tolerance | 40 % from median page | rejects stitch edge bands (C12) |
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

### Tests

32 tests, ~2.6 s. Unit tests for box geometry, folder lifecycle and size
filtering; end-to-end tests
end-to-end tests drive the real CLI against a generated 4×3 card.

---

## What is UNPROVEN

Do not assume any of this works. None of it has been exercised.

- ~~**Multi-card journals.**~~ **PROVEN 2026-08-22** — see below.
- **Real archival material.** Every run so far has been on *one* physical test
  card containing a Yamaha RD500LC service manual. Never run on real patient
  journals. Page density, contrast, and layout of real cards may differ.
- **`--order rows`.** Never used on real data. The test card is `columns`.
- **`--invert`.** Never exercised at all.
- **`--format jpg`.** Not run since today's changes. Forbidden for pipeline
  output anyway (contract C7), but the code path exists.
- **Zero-page failure in production.** Verified with a synthetic specks-only
  card, never on a real bad scan. Note the failure path *moves the operator's
  source file* — that behaviour has only ever run against throwaway inputs.
- **The `error/` folder location.** We write to `<input dir>/error/`, i.e.
  `Panoramas/error/`. There is also a separate `NHA/Error/` used by other stages,
  and C13's archive sits beside the input rather than inside it. None of these
  three agree and nobody has reconciled them.
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

- The header prepage is implemented but **off by default** pending Trond's "next
  round". Enable with `--header-page` when the import side is ready.
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
