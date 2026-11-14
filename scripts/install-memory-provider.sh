#!/usr/bin/env bash
# install-memory-provider.sh — attach wetnoodle to Hermes as a memory provider.
#
# The provider does two things the MCP tools cannot:
#   * pre-turn recall — the user's message is searched before the model sees it, and
#     the passages that clear a score floor are injected as context;
#   * a `memory_search` tool over the archive.
#
# It is a Hermes plugin, so it must live in a profile's plugins directory. This
# script symlinks the repository's copy there instead of copying it, so the profile
# never becomes a second, drifting source of truth.
#
#   ./scripts/install-memory-provider.sh                 # every profile found
#   ./scripts/install-memory-provider.sh work            # only these
#   ./scripts/install-memory-provider.sh --remove        # unlink from every profile
#
# Idempotent. Existing real directories are moved aside, never deleted.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN="$ROOT/plugin"
HERMES_HOME="${RAG_HERMES_HOME:-$HOME/.hermes}"
DB_DIR="${RAG_DIR:-$HOME/.wetnoodle}"

die() { echo "install-memory-provider: $*" >&2; exit 1; }

HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
if [ -z "$HERMES_BIN" ] && [ -x "$HOME/.local/bin/hermes" ]; then
    HERMES_BIN="$HOME/.local/bin/hermes"
fi
[ -n "$HERMES_BIN" ] || die "hermes not found — put it on PATH or set HERMES_BIN"

REMOVE=0
[ "${1:-}" = "--remove" ] && { REMOVE=1; shift; }

[ -f "$PLUGIN/__init__.py" ] || die "plugin/ not found in $ROOT — run this from the checkout"

profiles=()
if [ "$#" -gt 0 ]; then
    profiles=("$@")
else
    [ -d "$HERMES_HOME/profiles" ] || die "no profiles under $HERMES_HOME (pass profile names)"
    for d in "$HERMES_HOME"/profiles/*/; do [ -d "$d" ] && profiles+=("$(basename "$d")"); done
fi
[ "${#profiles[@]}" -gt 0 ] || die "no profiles found"

echo "wetnoodle memory provider"
echo "  repo    : $ROOT"
echo "  hermes  : $HERMES_HOME"
echo "  store   : $DB_DIR"
echo

for p in "${profiles[@]}"; do
    pdir="$HERMES_HOME/profiles/$p"
    [ -d "$pdir" ] || { echo "  $p: no such profile — skipped"; continue; }
    dest="$pdir/plugins/wetnoodle"

    if [ "$REMOVE" -eq 1 ]; then
        if [ -L "$dest" ]; then rm -f "$dest"; echo "  $p: unlinked"
        elif [ -d "$dest" ]; then echo "  $p: $dest is a real directory, left alone"
        else echo "  $p: not installed"; fi
        continue
    fi

    mkdir -p "$pdir/plugins"
    if [ -L "$dest" ]; then
        rm -f "$dest"
    elif [ -d "$dest" ]; then
        mv "$dest" "$dest.bak-$(date +%Y%m%d-%H%M%S)"
        echo "  $p: existing copy moved aside"
    fi
    ln -s "$PLUGIN" "$dest"

    # The provider reads these; without them it falls back to ~/.wetnoodle, which is
    # a different (usually smaller) archive than the one the MCP tools serve.
    # Do not swallow this: a linked plugin with no provider key looks installed and
    # does nothing, which is the failure mode this script exists to prevent.
    cfg_ok=1
    "$HERMES_BIN" -p "$p" config set memory.provider wetnoodle --force >/dev/null 2>&1 || cfg_ok=0
    "$HERMES_BIN" -p "$p" config set memory.wetnoodle.repo "$ROOT" --force >/dev/null 2>&1 || cfg_ok=0
    "$HERMES_BIN" -p "$p" config set memory.wetnoodle.db_dir "$DB_DIR" --force >/dev/null 2>&1 || cfg_ok=0
    if [ "$cfg_ok" -eq 1 ]; then
        echo "  $p: linked, provider=wetnoodle, store=$DB_DIR"
    else
        echo "  $p: linked, but the config keys were NOT set — run 'hermes -p $p memory status'"
    fi
done

echo
echo "Restart the gateway for the provider to load:  hermes -p <profile> gateway run"
echo "Check it landed:                              hermes -p <profile> memory status"
