"""Safe report extraction for the air-gapped production machine (m4-studio).

Inspects every panorama in a folder (--skip-extraction --anon-viz, touching
neither sources nor existing card folders) and collects ONLY anonymized
artifacts into a report folder for the USB stick: per-card text logs, the
anonymized visualizations, and a SAMMENDRAG.txt. Journal content
(visualization.jpg, binaries, page crops) must never end up there - not even
by accident, so every copy passes a hard whitelist and the finished folder is
re-scanned before it is announced.

Driven by RAPPORT.command (Finder double-click); testable from pytest.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from segment_microfiche import code_version

REPO = Path(__file__).resolve().parent
SEGMENTER = REPO / "segment_microfiche.py"
PANORAMA_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


class UnsafeArtifact(Exception):
    """Refused to copy a file whose name is not on the report whitelist."""


def is_safe_artifact(name):
    """Only files we generate ourselves, with no journal content in them."""
    return (name == "SAMMENDRAG.txt"
            or name == "anon_viz.jpg" or name.endswith("_anon_viz.jpg")
            or name.endswith("_rapport.txt"))


def copy_safe(src, dst):
    """The ONLY way files enter the report folder. Check before any mkdir so a
    refused copy leaves no trace."""
    dst = Path(dst)
    if not is_safe_artifact(dst.name):
        raise UnsafeArtifact(f"{dst.name} er ikke paa hvitelisten og skal "
                             "ikke ut av maskinen")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


QUALITY_WARN_BELOW = 50  # field data 2026-09-07: sick cards < 50, healthy > 74


QUALITY_POOR_BELOW = 60  # segment_microfiche grades below this as POOR


def parse_grid(output):
    m = re.search(r"Detected grid: (.+)", output)
    return m.group(1).strip() if m else None


def summary_line(stem, exit_code, pages, fragment_groups, anon_missing=False,
                 quality=None, grid=None):
    if anon_missing:
        # Whatever the exit code said: without the anonymized image the card
        # cannot be inspected across the air gap, and a missing expected
        # artifact must never read as success.
        return (f"FEIL      exit {exit_code}  {pages:3d} sider  {stem}  "
                "(anon_viz mangler)")
    warn = ""
    if quality is not None and quality < QUALITY_WARN_BELOW:
        warn = f"  ADVARSEL LAV KVALITET: {quality}"
    # Quality and grid on every row: card 111 shipped four pages in one box
    # at quality 52.5 and the summary just said OK (2026-09-08).
    detail = f"  kv {quality if quality is not None else '?':>5}"
    detail += f"  {grid or '?':<20}"
    if exit_code == 0:
        # A card the segmenter is not confident about must not read as fine.
        label = ("SVAK    " if quality is not None
                 and quality < QUALITY_POOR_BELOW else "OK      ")
        return f"{label}  exit 0  {pages:3d} sider{detail}  {stem}{warn}"
    if exit_code == 3:
        return (f"FRAGMENT  exit 3  {pages:3d} sider{detail}  {stem}  "
                f"({fragment_groups} grupper){warn}")
    return (f"FEIL      exit {exit_code}  {pages:3d} sider{detail}  "
            f"{stem}{warn}")


def find_panoramas(folder):
    """Images directly in the folder - never subfolders, which on a
    production disk hold card folders full of extracted journal pages."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in PANORAMA_SUFFIXES)


def inspect_panorama(panorama, workdir, extra_args=()):
    """One inspection run. Never touches the source (--skip-extraction) and
    writes only to workdir, which stays outside the report folder because it
    holds the NON-anonymized visualization. extra_args flow to the
    segmenter - the A/B path for flagged modes like --background-first."""
    proc = subprocess.run(
        [sys.executable, str(SEGMENTER),
         "-i", str(panorama), "-O", str(workdir),
         "--skip-extraction", "--anon-viz", *extra_args],
        capture_output=True, text=True, cwd=str(REPO))
    return proc.returncode, proc.stdout + proc.stderr


def count_pages(workdir):
    csv_path = Path(workdir) / "page_coordinates.csv"
    if not csv_path.exists():
        return 0
    lines = csv_path.read_text().splitlines()
    return max(0, len(lines) - 2)  # minus quality comment + column header


def count_fragment_groups(output):
    m = re.search(r"(\d+) suspected page fragment", output)
    return int(m.group(1)) if m else 0


def parse_quality(output):
    m = re.search(r"Card Quality: ([\d.]+)/100", output)
    return float(m.group(1)) if m else None


def unique_dir(base):
    base = Path(base)
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.name}-{n}")
    return candidate


def run_report(source_folder, report_dir, open_finder=True, extra_args=()):
    source_folder = Path(source_folder)
    panoramas = find_panoramas(source_folder)
    if not panoramas:
        raise SystemExit(
            f"FEIL: fant ingen panoramabilder direkte i {source_folder}\n"
            "Velg mappen som inneholder selve bildefilene "
            "(jpg/jpeg/tif/tiff/png).")

    report_dir = unique_dir(report_dir)
    report_dir.mkdir(parents=True)
    rows = []
    for panorama in panoramas:
        stem = panorama.stem
        print(f"Inspiserer {panorama.name} ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / stem
            try:
                exit_code, output = inspect_panorama(panorama, workdir,
                                                     extra_args)
            except Exception as exc:  # a crash must land in SAMMENDRAG, loudly
                exit_code, output = -1, f"KLARTE IKKE AA KJOERE: {exc!r}"
            (report_dir / f"{stem}_rapport.txt").write_text(output)
            pages = count_pages(workdir)
            anon = workdir / "_debug" / "anon_viz.jpg"
            if anon.exists():
                copy_safe(anon, report_dir / f"{stem}_anon_viz.jpg")
            rows.append(summary_line(stem, exit_code, pages,
                                     count_fragment_groups(output),
                                     anon_missing=not anon.exists(),
                                     quality=parse_quality(output),
                                     grid=parse_grid(output)))

    ok = sum(1 for r in rows if r.startswith("OK"))
    svak = sum(1 for r in rows if r.startswith("SVAK"))
    frag = sum(1 for r in rows if r.startswith("FRAGMENT"))
    fail = len(rows) - ok - svak - frag
    mode = ("bakgrunn-foerst" if "--background-first" in extra_args
            else "standard")
    summary = "\n".join([
        f"RAPPORT generert {date.today().isoformat()}",
        f"Kilde: {source_folder}",
        f"Kode: {code_version()}  |  Modus: {mode}",
        f"Kort: {len(rows)}  |  OK: {ok}  |  SVAK: {svak}  |  "
        f"FRAGMENTER: {frag}  |  FEIL: {fail}",
        "",
        *rows, ""])
    (report_dir / "SAMMENDRAG.txt").write_text(summary)

    # Defense in depth: nothing but whitelisted names may sit in the finished
    # report folder, no matter what the code above did.
    leaked = [p.name for p in report_dir.iterdir()
              if not is_safe_artifact(p.name)]
    if leaked:
        raise UnsafeArtifact(f"ikke-hvitelistede filer i rapportmappen: "
                             f"{leaked} - IKKE kopier den til minnepinnen")

    print(summary)
    print(f"Kopier mappen {report_dir.name} til minnepinnen")
    if open_finder:
        subprocess.run(["open", str(report_dir)])
    return report_dir


def main(argv):
    extra = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]
    if not positional:
        raise SystemExit("Bruk: rapport.py <mappe-med-panoramaer> "
                         "[rapportmappe] [--background-first ...]")
    default_report = (Path.home() / "Desktop"
                      / f"RAPPORT-{date.today().isoformat()}")
    report = Path(positional[1]) if len(positional) > 1 else default_report
    run_report(positional[0], report, extra_args=extra)


if __name__ == "__main__":
    main(sys.argv[1:])
