"""Lossless viewing copies of zstd-TIFF panoramas (m4-studio).

Preview cannot open the zstd-compressed panoramas; this writes an LZW copy
NEXT TO each source (`<navn>_visning.tif`, never overwriting) so Trond can
inspect them - including measuring real pixel values with Digital Color
Meter, which is why the copy must be lossless. Viewing copies contain
journal data and stay on the machine; they are never report artifacts.

Driven by VIS-PANORAMA.command (Finder double-click); testable from pytest.
"""

import subprocess
import sys
from pathlib import Path

import pyvips

TIFF_SUFFIXES = {".tif", ".tiff"}
VIEW_MARKER = "_visning"


def view_path(src):
    """<navn>_visning.tif beside the source; -2, -3... on collision."""
    src = Path(src)
    base = src.with_name(f"{src.stem}{VIEW_MARKER}.tif")
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = src.with_name(f"{src.stem}{VIEW_MARKER}-{n}.tif")
    return candidate


def sources(args):
    """Each argument is a TIFF file or a folder; folders contribute only
    their DIRECT children - subfolders on the production disk hold card
    folders with extracted journal pages."""
    out = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out.extend(sorted(
                c for c in p.iterdir()
                if c.is_file() and c.suffix.lower() in TIFF_SUFFIXES
                and VIEW_MARKER not in c.stem))
        elif p.is_file() and p.suffix.lower() in TIFF_SUFFIXES:
            out.append(p)
        else:
            raise SystemExit(f"FEIL: {p} er hverken en .tif/.tiff eller en "
                             "mappe")
    return out


def convert(src):
    """Write the lossless LZW viewing copy; returns its path."""
    dst = view_path(src)
    img = pyvips.Image.new_from_file(str(src), access='sequential')
    img.write_to_file(str(dst), compression='lzw', bigtiff=True)
    return dst


def main(argv):
    if not argv:
        raise SystemExit("Bruk: vis_panorama.py <tif-fil eller mappe> ...")
    todo = sources(argv)
    if not todo:
        raise SystemExit("FEIL: fant ingen .tif/.tiff aa konvertere")
    folders = []
    for src in todo:
        print(f"Konverterer {src.name} ...", flush=True)
        dst = convert(src)
        print(f"  -> {dst.name}")
        if dst.parent not in folders:
            folders.append(dst.parent)
    print("\nMERK: visningskopiene inneholder journaldata og blir paa "
          "denne maskinen - de skal ALDRI ut paa minnepinne.")
    for folder in folders:
        subprocess.run(["open", str(folder)])


if __name__ == "__main__":
    main(sys.argv[1:])
