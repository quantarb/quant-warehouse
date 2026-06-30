from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def recipe_hash(
    *,
    sections: Sequence[str],
    providers: Sequence[str],
    transforms: Mapping[str, Any] | None = None,
    length: int = 16,
) -> str:
    """Stable hash for a feature materialization recipe."""
    payload = {
        "sections": sorted({str(s) for s in sections}),
        "providers": sorted({str(p) for p in providers}),
        "transforms": transforms or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return digest[:length]
