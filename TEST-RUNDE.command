#!/bin/zsh
# TEST-RUNDE.command - dobbeltklikkes i Finder paa m4-studio.
#
# Kjoerer en hel testrunde fra en liste med kort-ID-er: finner hvert
# panorama (Panoramas, Panoramas/error, PanoramaArchive, Error), KOPIERER
# det til en fersk testmappe - flytter aldri, roerer aldri kildene - og
# inspiserer kopiene i baade standard og bakgrunn-foerst. Til slutt
# SAMMENLIGNING.txt mot forrige testrunde, med trappeloepet og
# celle-beviset per kort.
#
# Kortlista: TEST-KORT.txt ved siden av dette skriptet, en ID per linje
# (# er kommentar). Finnes den ikke, spoer den etter en fil.
#
# Finder/launchd gir IKKE vanlig PATH: absolutte stier utledet fra
# skriptets egen plassering, aldri naken python3.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "FEIL: fant ikke $PY"
    echo "Kjoer OPPDATER.command foerst slik at .venv er paa plass."
    exit 1
fi

exec "$PY" "$SCRIPT_DIR/test_runde.py" "$@"
