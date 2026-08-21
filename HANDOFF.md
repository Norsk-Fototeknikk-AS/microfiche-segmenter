# Handoff — state as of 2026-08-21

Written before the machine goes offline for production. Read `README.md` first;
its "Contracts" section is the part that must not be broken. This file records
what was measured, what is still unproven, and how to check your work with no
network and no access to the conversations that produced this code.

Everything below was measured on this machine on 2026-08-21, not estimated.

---

## What changed today

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

## Measured numbers

### Production card `612130000012_00016`

```
input    /Volumes/NB02/NHA/Panoramas/612130000012_00016.tif
         34354 × 25533 (877 MP), 1.14 GB, 3 bands
output   /Volumes/NB02/NHA/Microfiche/612130000012_00016/
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

### Constants that matter

| thing | value | full-res equivalent |
|---|---|---|
| detect scale | 0.1 | — |
| detect erosion | 7×7, 2 iterations | 6 px → **60 px** per side |
| refine local scale | 0.2 | — |
| refine erosion | 3×3, 2 iterations | 2 px → **10 px** per side |
| default margin | 1 % of median page | 20 × 16 px on the production card |
| extraction workers | 5 | — |

`--refine` on the production-grade card reports avg shift 6 px x / 11 px y from
the global pass, i.e. the two passes now agree. Before the erosion fix they
disagreed by ~50 px. Refine is slightly *looser* than the default path
(−49 px vs −10 px on the left edge of page 1) — both safe, default is tighter.

### Tests

22 tests, ~1.6 s. Unit tests for box geometry and folder lifecycle; four
end-to-end tests drive the real CLI against a generated 4×3 card.

---

## What is UNPROVEN

Do not assume any of this works. None of it has been exercised.

- **Multi-card journals.** No journal has ever spanned two cards here. The
  `<fid>_00032` naming for card 2 is agreed but has never been produced,
  segmented, or imported. Grouping and ordering of two cards under one
  fanearkID is untested end to end.
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
  `Panoramas/error/`. There is also a separate `/Volumes/NB02/NHA/Error/` used
  by other stages. These are not the same folder and nobody has reconciled them.
- **Output root.** `/Volumes/NB02/NHA/Microfiche/` was chosen by inference —
  it was empty, purpose-named, and created alongside `Error/`.
  `~/.ocr-pipeline-config.json` does not exist, so `segmenterWatchRoot` is still
  `""` and there was no authoritative value to read. **Confirm this before
  trusting it.** If it is wrong the fix is a folder move, not a re-run.
- **header.json / header_proxy.** We do not produce them. They are optional
  downstream, but if they ever become required, nothing here writes them.

---

## Pitfalls

- **A green test suite does not prove the geometry is right.** The erosion bug
  passed every test that existed. Geometry changes must be verified by measuring
  crops against the source image (recipe below).
- **stdout is block buffered.** No explicit flushing anywhere. Run with
  `python -u` or `PYTHONUNBUFFERED=1` if anything parses progress output.
- **`-O` is the card folder, not the root.** Easy to get wrong; produces a
  correct-looking folder in the wrong place.
- **The quality score is information, not a gate.** A card scoring 30/100 is
  processed exactly like one scoring 95/100. Only *zero pages* fails.
- **Detection never sees a barcode.** fanearkIDs come from the Capture One
  AppleScript chain (`~/Projects/CaptureOneNamer`, counter over `names.txt`),
  not from anything on the card. If IDs look wrong, the cause is upstream —
  no rule applied to filenames here can recover a correct ID.

---

## Verifying a change offline

No network needed for any of this.

**1. Run the suite.**

```bash
.venv/bin/python -m pytest test_segment.py -q      # expect 22 passed
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

- Nothing in this repo was committed today. `segment_microfiche.py`,
  `requirements.txt`, `README.md`, `HANDOFF.md` and the new `test_segment.py`
  are all uncommitted working-tree changes.
- The pre-fix output folder on the Desktop
  (`~/Desktop/NHA Mikrofiche/Output/segmented/612130000333_0000736115 Panorama/`)
  still holds coordinates from before the erosion fix, plus a `_done` that the
  OCR app may act on. It was left alone deliberately — unknown whether the app
  already consumed it.
- Confirm the output root (see UNPROVEN).
