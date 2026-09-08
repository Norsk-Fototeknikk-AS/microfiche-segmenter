"""One orchestrated test round, from a list of card IDs to a comparison.

The manual round cost an evening: panoramas moved by hand between
Panoramas, error and PanoramaArchive, the app's `_done` rules to mind, logs
copied one at a time. This runs the round instead.

It **copies** the named cards into a fresh folder and never moves anything,
so a test round can never disturb the production queue - the sources stay
exactly where the app and the runner expect them. Then it inspects the copy
in both modes and diffs the result against the previous round.

THE RULE: only reports and anonymized artifacts leave this machine. The
panorama copies are journal data and stay here, so they are written to a
SEPARATE folder outside the report tree (`TEST-PANORAMAER-<date>/`, with a
`LES-MEG-IKKE-KOPIER.txt` inside saying so). The report folder holds nothing
but the two report folders and SAMMENLIGNING.txt, so "copy the test round
folder to the stick" is a safe sentence.

That separation is for the HUMAN. The machine was already safe - every copy
goes through `copy_out`, which refuses anything outside the whitelist - but
in the first version the panoramas sat beside the reports, and no whitelist
helps against someone dragging the whole folder (Trond, 2026-09-08).
"""

import json
import re
import shutil
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

import rapport

REPO = Path(__file__).resolve().parent
CONFIG = Path.home() / ".microfiche-station.json"
DEFAULT_SESSION_ROOT = "/Users/m4-studio/Desktop/NHA"
PANORAMA_SUFFIXES = rapport.PANORAMA_SUFFIXES

# Where a card can be. Three of these are the segmenter's own (C9, C13) and
# the fourth is the app's `errorDir` - HANDOFF records that the three error
# locations have never been reconciled, so a card really can be in any of
# them. Searching all four makes that visible instead of reporting the card
# as missing.
SOURCE_FOLDERS = ("Panoramas", "Panoramas/error", "PanoramaArchive", "Error")

USB_REPORT_DIR = Path("/Volumes/Samsung 256GB/Mikrofiche/Rapport")


def is_safe_artifact(name):
    """rapport.py's whitelist plus this tool's one addition. Composed, not
    copied: the whitelist keeps a single source of truth."""
    return rapport.is_safe_artifact(name) or name == "SAMMENLIGNING.txt"


def session_root():
    """From ~/.microfiche-station.json like the app, else the default."""
    try:
        cfg = json.loads(CONFIG.read_text())
        return Path(cfg["sessionRoot"])
    except (OSError, ValueError, KeyError):
        return Path(DEFAULT_SESSION_ROOT)


def read_card_ids(path):
    """One card ID per line. Blank lines and # comments are ignored."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def find_panoramas(card_ids, root):
    """([(path, which folder)], [ids not found anywhere])."""
    found, missing = [], []
    for cid in card_ids:
        hit = None
        for sub in SOURCE_FOLDERS:
            folder = Path(root) / sub
            if not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if (p.is_file() and p.suffix.lower() in PANORAMA_SUFFIXES
                        and p.stem == cid):
                    hit = (p, sub)
                    break
            if hit:
                break
        if hit:
            found.append(hit)
        else:
            missing.append(cid)
    return found, missing


def stage_cards(card_ids, root, dest):
    """Copy every named panorama into dest. Never moves, never touches the
    source - a test round must not disturb the production queue."""
    found, missing = find_panoramas(card_ids, root)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for path, sub in found:
        shutil.copyfile(path, dest / path.name)
        print(f"  {path.name}  <- {sub}")
    return len(found), missing


def _parse_report(folder):
    """{stem: dict} from a report folder written by rapport.py."""
    folder = Path(folder)
    cards = {}
    summary = folder / "SAMMENDRAG.txt"
    rows = summary.read_text().splitlines() if summary.exists() else []
    for line in rows:
        m = re.match(r"^(\w+)\s+exit (\d+)\s+(\d+) sider", line)
        if not m:
            continue
        stem = line.split()[-1]
        cards[stem] = {"status": m.group(1), "exit": int(m.group(2)),
                       "pages": int(m.group(3)), "grid": "?", "quality": None,
                       "step2": "TRINN2" in line, "cells": []}
    for f in sorted(folder.glob("*_rapport.txt")):
        stem = f.name[:-len("_rapport.txt")]
        text = f.read_text(errors="replace")
        card = cards.setdefault(stem, {"status": "?", "exit": -1, "pages": 0,
                                       "grid": "?", "quality": None,
                                       "step2": False, "cells": []})
        g = re.search(r"Detected grid: (.+)", text)
        q = re.search(r"Card Quality: ([\d.]+)", text)
        if g:
            card["grid"] = g.group(1).strip()
        if q:
            card["quality"] = float(q.group(1))
        card["step2"] = card["step2"] or "Step 2 threshold:" in text
        # What the staircase did, for the line Trond reads first.
        m = re.search(r"Step 2 threshold: trigger (.+?), border (\d+)% "
                      r"\(over \d+%\), otsu (\d+) -> (\w+)", text)
        card["step2_detail"] = None
        if m:
            after_m = re.search(r"Step 2 proved itself: border \d+% -> "
                                r"(\d+)%", text)
            failed = "re-threshold failed" in text
            fail_m = re.search(r"border (\d+)% -> (\d+)%", text)
            card["step2_detail"] = {
                "trigger": m.group(1), "border_before": m.group(2),
                "otsu_before": m.group(3), "otsu_after": m.group(4),
                "border_after": (after_m.group(1) if after_m else
                                 (fail_m.group(2) if fail_m else "?")),
                "outcome": "MISLYKTES" if failed else
                           ("bevist" if after_m else "ukjent")}
        card["cells"] = [(int(p), float(fg)) for p, fg in
                         re.findall(r"CELL .* page=([01]) fg=([\d.]+)", text)]
    return cards


def _describe(card):
    return (f"exit {card['exit']} {card['pages']:3d} sider  "
            f"kv {card['quality'] if card['quality'] is not None else '?':>5}  "
            f"{card['grid']:<24}{'  TRINN2' if card['step2'] else ''}")


def _cell_line(stem, card):
    parts = []
    for page in (1, 0):
        vals = [fg for p, fg in card["cells"] if p == page]
        if vals:
            parts.append(f"page={page} n={len(vals)} "
                         f"fg_median={statistics.median(vals):.2f}")
    if not parts:
        return None
    return f"  CELL {stem}  " + "  |  ".join(parts)


def comparison(after_dir, before_dir):
    """SAMMENLIGNING.txt: this round against the previous one, differences
    marked in the first column so they can be found by eye and by grep."""
    now = _parse_report(after_dir)
    before = _parse_report(before_dir) if before_dir else {}
    out = [f"SAMMENLIGNING {date.today().isoformat()}",
           f"Naa:    {after_dir}",
           f"Foer:   {before_dir if before_dir else '(ingen tidligere runde)'}",
           "",
           "= uendret   ! endret   + nytt kort   - borte",
           "TRINN2-linja viser trappeloepet: trigger, otsu og border foer/etter",
           ""]
    for stem in sorted(set(now) | set(before)):
        if stem not in now:
            out.append(f"- {stem}  {_describe(before[stem])}   (borte)")
            continue
        if stem not in before:
            out.append(f"+ {stem}  {_describe(now[stem])}")
        else:
            a, b = now[stem], before[stem]
            same = all(a[k] == b[k] for k in
                       ("exit", "pages", "grid", "quality", "step2"))
            if same:
                out.append(f"= {stem}  {_describe(a)}")
            else:
                # Both states on ONE line: a change must be greppable, not
                # just visible to a reader scrolling past it.
                out.append(f"! {stem}  {_describe(a)}   (foer: "
                           f"{_describe(b).strip()})")
        d = now[stem].get("step2_detail")
        if d:
            out.append(f"  TRINN2: {stem}  {d['outcome']}  "
                       f"trigger {d['trigger']}  otsu {d['otsu_before']} -> "
                       f"{d['otsu_after']}  border {d['border_before']}% -> "
                       f"{d['border_after']}%")
        line = _cell_line(stem, now[stem])
        if line:
            out.append(line)
    return "\n".join(out) + "\n"


def previous_round(parent, current):
    """The newest TEST-RUNDE folder before this one, if any."""
    rounds = sorted(p for p in Path(parent).glob("TEST-RUNDE-*")
                    if p.is_dir() and p != current)
    for r in reversed(rounds):
        std = r / "rapport-standard"
        if (std / "SAMMENDRAG.txt").exists():
            return std
    return None


def copy_out(files, dest):
    """Whitelisted copies to the stick, each verified byte for byte."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        if not is_safe_artifact(src.name):
            raise rapport.UnsafeArtifact(
                f"{src.name} er ikke paa hvitelisten og skal ikke ut av "
                "maskinen")
        target = dest / src.name
        shutil.copyfile(src, target)
        if src.read_bytes() != target.read_bytes():
            raise OSError(f"kopien av {src.name} stemmer ikke med kilden")
    return dest


def run_round(card_file, parent=None, open_finder=True):
    parent = Path(parent) if parent else Path.home() / "Desktop"
    card_ids = read_card_ids(card_file)
    if not card_ids:
        raise SystemExit(f"FEIL: ingen kort-ID-er i {card_file}")
    root = session_root()
    today = date.today().isoformat()
    folder = rapport.unique_dir(parent / f"TEST-RUNDE-{today}")
    folder.mkdir(parents=True)
    # Journal data stays on the machine, and stays OUT of the folder anyone
    # might carry to the stick.
    panoramas = rapport.unique_dir(parent / f"TEST-PANORAMAER-{today}")
    panoramas.mkdir(parents=True)
    (panoramas / "LES-MEG-IKKE-KOPIER.txt").write_text(
        "Denne mappen inneholder KOPIER AV PANORAMAENE - journaldata.\n"
        "Den skal BLI PAA MASKINEN og aldri paa minnepinnen.\n\n"
        "Bare rapporter og anonymiserte artefakter forlater m4-studio.\n"
        f"Rapportene fra denne runden ligger i {folder.name}/ - den mappen\n"
        "er trygg aa kopiere i sin helhet.\n\n"
        "Naar runden er lest kan denne mappen slettes.\n")
    print(f"Testrunde i {folder}\nPanoramakopier (BLIR PAA MASKINEN): "
          f"{panoramas}\nKilde: {root}\n")

    copied, missing = stage_cards(card_ids, root, panoramas)
    if missing:
        print(f"\nFANT IKKE {len(missing)} kort (soekte i "
              f"{', '.join(SOURCE_FOLDERS)}):", file=sys.stderr)
        for cid in missing:
            print(f"  {cid}", file=sys.stderr)
    if not copied:
        raise SystemExit("FEIL: ingen av kortene ble funnet - ingenting aa kjoere")
    print(f"\n{copied} kort kopiert (kildene er uroert)\n")

    reports = {}
    for label, extra in (("standard", ()), ("bakgrunn", ("--background-first",))):
        print(f"=== Kjoerer {label} ===")
        reports[label] = rapport.run_report(
            panoramas, folder / f"rapport-{label}",
            open_finder=False, extra_args=extra)

    text = comparison(reports["standard"], previous_round(parent, folder))
    (folder / "SAMMENLIGNING.txt").write_text(text)
    print("\n" + text)

    if USB_REPORT_DIR.parent.is_dir():
        out = copy_out([folder / "SAMMENLIGNING.txt"],
                       USB_REPORT_DIR / folder.name)
        for label in reports:
            for f in sorted(reports[label].iterdir()):
                copy_out([f], USB_REPORT_DIR / folder.name / f"rapport-{label}")
        print(f"Kopiert til pinnen: {out}")
    else:
        print(f"Pinnen er ikke montert - rapportene ligger i {folder}")
    if open_finder:
        subprocess.run(["open", str(folder)])
    return folder


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if args:
        card_file = args[0]
    else:
        default = REPO / "TEST-KORT.txt"
        if default.exists():
            card_file = default
        else:
            card_file = subprocess.run(
                ["osascript", "-e",
                 'POSIX path of (choose file with prompt "Velg tekstfil med '
                 'kort-ID-er, en per linje:")'],
                capture_output=True, text=True).stdout.strip()
            if not card_file:
                raise SystemExit("Avbrutt.")
    run_round(card_file, parent=Path(args[1]) if len(args) > 1 else None)


if __name__ == "__main__":
    main(sys.argv[1:])
