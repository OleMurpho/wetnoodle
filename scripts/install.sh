#!/usr/bin/env bash
# install.sh — put wetnoodle on a Hermes agent that already has Ollama.
#
# One command, in the order that matters, on any host:
#
#   ./scripts/install.sh                        # this profile, default scope
#   ./scripts/install.sh -p mnemosyne -p robert --scope-shared
#   ./scripts/install.sh -p ordo --scope-shared --deliver discord:123456789
#
# It runs, in order:
#   1. bootstrap.sh        venv, dependencies, local Ollama + the embed model, doctor, first ingest
#   2. register-mcp.sh     the MCP server, with scope baked into the profile's config
#   3. install-memory-provider.sh   pre-turn recall, with the same scope
#   4. install-refresh-job.sh       the watchdog cron job, silent unless it fails
#
# Every step is idempotent, so re-running this is how you repair an install.
# Nothing here is specific to one machine: paths come from the checkout, the
# store comes from the profile's own configuration, and the scope is the flag
# you pass.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
[ -z "$HERMES_BIN" ] && [ -x "$HOME/.local/bin/hermes" ] && HERMES_BIN="$HOME/.local/bin/hermes"
HERMES_HOME="${RAG_HERMES_HOME:-$HOME/.hermes}"

PROFILES=()
SCOPE_ARGS=()
DELIVER=""
SKIP_BOOTSTRAP=0

die() { echo "error: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        -p|--profile) [ "$#" -ge 2 ] || die "$1 needs a profile"; PROFILES+=("$2"); shift 2 ;;
        --profile=*) PROFILES+=("${1#*=}"); shift ;;
        --scope-shared) SCOPE_ARGS=(--scope-shared); shift ;;
        --scope) SCOPE_ARGS=(--scope); shift ;;
        --allow) [ "$#" -ge 2 ] || die "--allow needs a value"; SCOPE_ARGS=(--allow "$2"); shift 2 ;;
        --allow=*) SCOPE_ARGS=(--allow "${1#*=}"); shift ;;
        --deliver) [ "$#" -ge 2 ] || die "--deliver needs a target"; DELIVER="$2"; shift 2 ;;
        --deliver=*) DELIVER="${1#*=}"; shift ;;
        --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[ "${#PROFILES[@]}" -eq 0 ] && PROFILES=("${HERMES_PROFILE:-default}")

echo "wetnoodle install"
echo "  checkout: $ROOT"
echo "  profiles: ${PROFILES[*]}"
echo "  scope   : ${SCOPE_ARGS[*]:-open (every corpus visible; use --scope-shared on a shared machine)}"

# Several profiles on one store with no scope is the configuration that lets agents read each
# other's transcripts, and it is the one an unattended install will silently produce.
existing=0
if [ -d "$HERMES_HOME/profiles" ]; then
    existing=$(find "$HERMES_HOME/profiles" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
fi
# Installing one profile at a time is how a store ends up shared without anyone deciding it:
# count what exists, not just what this invocation names.
if [ $(( existing + ${#PROFILES[@]} )) -gt 1 ] && [ "${#SCOPE_ARGS[@]}" -eq 0 ]; then
    die "this store would end up serving $(( existing + ${#PROFILES[@]} )) profiles with no scope,
       which shares every transcript between them.
       Pass --scope-shared (hide the other profiles, keep skills and imports) or
       --scope (each profile sees only its own corpus), or install one profile at a time."
fi

if [ "$SKIP_BOOTSTRAP" -eq 0 ]; then
    echo
    echo "== bootstrap =="
    "$ROOT/scripts/bootstrap.sh" || die "bootstrap failed — fix what it printed and re-run"
fi

[ -x "$ROOT/.venv/bin/python" ] || die "no interpreter at $ROOT/.venv/bin/python"

echo
echo "== MCP server =="
# ${arr[@]+...} is the portable idiom: bash 3.2 (macOS) errors on an empty array under set -u,
# and the no-scope install is the common case.
"$ROOT/scripts/register-mcp.sh" "${SCOPE_ARGS[@]+"${SCOPE_ARGS[@]}"}" "${PROFILES[@]}" || die "register-mcp failed"

echo
echo "== memory provider =="
"$ROOT/scripts/install-memory-provider.sh" "${SCOPE_ARGS[@]+"${SCOPE_ARGS[@]}"}" "${PROFILES[@]}" || die "install-memory-provider failed"

# One watchdog per store, not one per profile: every job ingests every corpus, so a second job
# on the same archive is two processes contending for the same lock for no extra coverage.
echo
echo "== refresh job =="
job_args=(-p "${PROFILES[0]}")
[ -n "$DELIVER" ] && job_args+=(--deliver "$DELIVER")
"$ROOT/scripts/install-refresh-job.sh" "${job_args[@]}" || die "install-refresh-job failed"

cat <<'DONE'

Nothing is live until the gateway restarts; that is deliberately not done here.
  systemctl --user restart hermes-gateway-<profile>     # or: hermes -p <profile> gateway run
Then check it:
  hermes -p <profile> memory status
  hermes -p <profile> cron list
DONE
