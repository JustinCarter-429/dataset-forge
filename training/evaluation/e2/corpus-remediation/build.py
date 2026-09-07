"""Stable entry point for the Phase E2.1 corpus builder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from corpus_remediation.builder import main

if __name__ == "__main__":
    raise SystemExit(main())
