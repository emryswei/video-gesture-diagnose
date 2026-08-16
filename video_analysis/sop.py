from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .models import SOPDefinition


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=8)
def load_sop(sop_id: str = "hk_chp_handrub") -> SOPDefinition:
    if re.fullmatch(r"[a-z0-9_]+", sop_id) is None:
        raise ValueError(f"Unknown SOP: {sop_id}")
    path = PROJECT_ROOT / "sops" / f"{sop_id}.json"
    if not path.is_file():
        raise ValueError(f"Unknown SOP: {sop_id}")
    return SOPDefinition.model_validate_json(path.read_text(encoding="utf-8"))
