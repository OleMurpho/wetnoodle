"""Locate and import the checkout's rag_core, wherever it lives.

Two layouts exist: modules at the checkout root (older) and under the `wetnoodle/`
package (current). Both are supported.

It is loaded **by file path under its own module name**, not as `wetnoodle.rag_core`.
Hermes loads this plugin from a directory whose name is also `wetnoodle`, so when the
loader registers it as `wetnoodle`, `from wetnoodle import rag_core` resolves to the
plugin package -- which has no rag_core -- and every availability check fails with a
misleading ImportError. rag_core is self-contained (no relative imports), so a path
load is safe and independent of what the loader calls us.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_NAME = "wetnoodle_rag_core"


def core_path(repo: Path | str) -> Path:
    """The rag_core.py inside *repo*, in whichever layout it uses."""
    repo = Path(repo)
    packaged = repo / "wetnoodle" / "rag_core.py"
    return packaged if packaged.exists() else repo / "rag_core.py"


def load_core(repo: Path | str):
    """Import and return the checkout's rag_core module (cached per process)."""
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    path = core_path(repo)
    if not path.exists():
        raise ModuleNotFoundError(f"rag_core.py not found under {Path(repo)}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)   # never leave a half-built module cached
        raise
    return module
