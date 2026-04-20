"""Root conftest — ensures the repo root is on sys.path for tests that
import ``physics_sim`` directly (rather than reading files)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
