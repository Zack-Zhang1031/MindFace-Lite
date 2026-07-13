from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_compat_command(prefix: Sequence[str]) -> None:
    from mindface.cli import main

    main([*prefix, *sys.argv[1:]])
