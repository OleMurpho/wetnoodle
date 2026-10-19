#!/usr/bin/env bash
# hygiene.sh — the pre-push gate. Run before every commit that touches docs or examples.
# Fails if machine-specific detail has leaked into a repo that is meant to be public, then checks that no
# data file is tracked. LICENSE is exempt: a copyright holder line is the one place a name belongs.
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERN='olemurpho|mlope|stormhold|stormrider|stormbreaker|slartibartfast|mnemosyne|archimedes|quizforge|192\.168\.|10\.255\.|/home/[a-z]+/|ghp_[A-Za-z0-9]{20,}'
# The canonical clone URL contains the owner's handle by necessity; that one line is whitelisted, and
# nothing else is. Anything else matching the pattern is a leak from the machine this was built on.
# Whitelisted, exactly two strings: the repo's own clone URL and its own Pages address. Both contain the
# owner's handle by necessity. Nothing else is exempt.
hits=$(grep -rniE "$PATTERN" . --exclude-dir=.git --exclude-dir=__pycache__ --exclude=LICENSE \
         --exclude=hygiene.sh \
       | grep -viE 'github\.com/OleMurpho/wetnoodle|olemurpho\.github\.io/wetnoodle' || true)
if [ -n "$hits" ]; then
  echo "FAIL — machine-specific strings in a public repo:"; echo "$hits"; exit 1
fi
echo "ok   no machine-specific strings"

if git rev-parse --git-dir >/dev/null 2>&1; then
  data=$(git ls-files | grep -E '\.db$|\.db-wal$|\.jsonl$|questions\.local\.json$' || true)
  if [ -n "$data" ]; then echo "FAIL — data files are tracked:"; echo "$data"; exit 1; fi
  echo "ok   no data files tracked"
fi

# LICENSE must exist and be a real free licence
if ! grep -q "MIT License" LICENSE 2>/dev/null; then echo "FAIL — LICENSE is not the expected MIT text"; exit 1; fi
echo "ok   MIT licence present"
