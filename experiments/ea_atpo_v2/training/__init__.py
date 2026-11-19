"""Offline Q regression, policy distillation, and evaluation for EA-ATPO V2."""

from pathlib import Path
import sys

V2_DIR = Path(__file__).resolve().parents[1]
V2_SRC = V2_DIR / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from v2_protocol import ACTION_ORDER  # noqa: E402

ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
