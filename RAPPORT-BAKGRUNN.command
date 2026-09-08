#!/bin/zsh
# RAPPORT-BAKGRUNN.command - B-siden av A/B-testen, dobbeltklikkes i Finder.
#
# Samme som RAPPORT.command, men segmenteren kjoerer i bakgrunn-foerst-modus
# (--background-first): jakken er den stabile klassen, sidene er avviket fra
# den. Rapportmappen faar "Modus: bakgrunn-foerst" i SAMMENDRAG og
# "mode bakgrunn-foerst" i hver rapport.txt, saa de to sidene aldri kan
# forveksles.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    SRC="$(osascript -e 'POSIX path of (choose folder with prompt "Velg mappen med panoramabildene (BAKGRUNN-FOERST-kjoering):")')"
else
    shift
fi
exec "$SCRIPT_DIR/RAPPORT.command" "$SRC" --background-first "$@"
