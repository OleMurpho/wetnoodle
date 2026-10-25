#!/usr/bin/env bash
# hygiene.sh — the pre-push gate. Run before every commit that touches docs or examples.
# Fails if machine-specific detail has leaked into a repo that is meant to be public, then checks that no
# data file is tracked. LICENSE is exempt: a copyright holder line is the one place a name belongs.
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERN='olemurpho|mlope|stormhold|stormrider|stormbreaker|slartibartfast|mnemosyne|archimedes|quizforge|192\.168\.|10\.255\.|/home/[a-z]+/|ghp_[A-Za-z0-9]{20,}'
# The canonical clone URL contains the owner's handle by necessity; that one line is whitelisted, and
# nothing else is. Anything else matching the pattern is a leak from the machine this was built on.
# Whitelisted, exactly three strings: the repo's own clone URL, its own Pages address, and the upstream
# project credited in README.md ("mnemosyne" is also an agent name on the machine this was built on, and
# the credit must survive the gate). Each contains a matching string by necessity. Nothing else is exempt.
# Scan only what would actually be published: the tracked files. Scanning the working tree means a local
# .venv trips the gate on its own absolute shebangs — which says nothing about what a reader of the repo sees.
if git rev-parse --git-dir >/dev/null 2>&1; then
  hits=$(git ls-files -z | xargs -0 -r grep -niE "$PATTERN" --exclude=LICENSE --exclude=hygiene.sh 2>/dev/null \
         | grep -viE 'github\.com/OleMurpho/wetnoodle|olemurpho\.github\.io/wetnoodle|mnemosyne-oss/mnemosyne' || true)
else
  hits=$(grep -rniE "$PATTERN" . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
           --exclude-dir=imports --exclude='*.db*' --exclude=LICENSE --exclude=hygiene.sh \
         | grep -viE 'github\.com/OleMurpho/wetnoodle|olemurpho\.github\.io/wetnoodle|mnemosyne-oss/mnemosyne' || true)
fi
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
