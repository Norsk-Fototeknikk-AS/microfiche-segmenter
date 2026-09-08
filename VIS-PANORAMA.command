#!/bin/zsh
# VIS-PANORAMA.command - dobbeltklikkes i Finder paa m4-studio.
#
# Lager TAPSFRIE visningskopier (<navn>_visning.tif, LZW) VED SIDEN AV
# zstd-TIFF-panoramaene som Forhaandsvisning ikke aapner, uten aa overskrive
# noe. Digital Color Meter paa kopien viser kildens ekte pikselverdier.
#
# MERK: visningskopiene inneholder journaldata og blir paa denne maskinen -
# de skal ALDRI ut paa minnepinne.
#
# Finder/launchd gir IKKE vanlig PATH: absolutte stier utledet fra skriptets
# egen plassering, aldri naken python3.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "FEIL: fant ikke $PY"
    echo "Kjoer OPPDATER.command foerst slik at .venv er paa plass."
    exit 1
fi

if (( $# > 0 )); then
    exec "$PY" "$SCRIPT_DIR/vis_panorama.py" "$@"
fi

# Én fil, eller Avbryt for aa velge en hel mappe i neste dialog.
SRC="$(osascript -e 'POSIX path of (choose file with prompt "Velg panorama-TIFF (Avbryt for aa velge en hel mappe i stedet)" of type {"public.tiff"})' 2>/dev/null)" || SRC=""
if [[ -z "$SRC" ]]; then
    SRC="$(osascript -e 'POSIX path of (choose folder with prompt "Velg mappen med panorama-TIFF-ene:")')"
fi

exec "$PY" "$SCRIPT_DIR/vis_panorama.py" "$SRC"
