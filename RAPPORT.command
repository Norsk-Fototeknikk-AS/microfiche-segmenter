#!/bin/zsh
# RAPPORT.command - dobbeltklikkes i Finder paa m4-studio.
#
# Inspiserer alle panoramabilder i en valgt mappe (roerer aldri kilder eller
# eksisterende sider) og samler KUN trygge, anonymiserte artefakter i
# ~/Desktop/RAPPORT-<dato>/: tekstrapporter, anon_viz.jpg per kort og en
# SAMMENDRAG.txt. Journalinnhold (visualization.jpg, binaerbilder, sideutsnitt)
# slipper aldri inn i rapportmappen - rapport.py haandhever en hviteliste.
#
# Finder/launchd gir IKKE vanlig PATH: bruk absolutte stier utledet fra
# skriptets egen plassering, aldri naken python3.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "FEIL: fant ikke $PY"
    echo "Kjoer OPPDATER.command foerst slik at .venv er paa plass."
    exit 1
fi

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    SRC="$(osascript -e 'POSIX path of (choose folder with prompt "Velg mappen med panoramabildene (arkiv- eller utdatamappen):")')"
else
    shift
fi

# Alt etter kildemappen gaar videre til rapport.py (f.eks. --background-first
# for A/B). Uten dette ble en flagget kjoering stille en standardkjoering -
# begge rapportsettene 2026-09-08 var standardmodus.
exec "$PY" "$SCRIPT_DIR/rapport.py" "$SRC" "$@"
