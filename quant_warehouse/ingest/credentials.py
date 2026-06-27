from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_QUANT_WAREHOUSE_HOME = Path(__file__).resolve().parents[2]


def _dotenv_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    qw_home = Path(os.environ.get("QW_HOME", str(_QUANT_WAREHOUSE_HOME))).expanduser()
    candidates.append(qw_home / ".env")
    candidates.append(_QUANT_WAREHOUSE_HOME / ".env")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return tuple(ordered)


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=path, override=False)
        return
    except Exception:
        pass

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = str(key).strip()
            if not key or key in os.environ:
                continue
            cleaned = str(value).strip().strip('"').strip("'")
            os.environ[key] = cleaned
    except Exception:
        return


def load_shared_env() -> None:
    """Load credentials from quant-warehouse .env files (best effort)."""
    for path in _dotenv_candidates():
        _load_dotenv_file(path)


@lru_cache(maxsize=1)
def resolve_fmp_api_key(*, required: bool = False) -> str:
    load_shared_env()
    api_key = str(os.environ.get("FMP_API_KEY") or "").strip()
    if required and not api_key:
        searched = ", ".join(str(path) for path in _dotenv_candidates())
        raise ValueError(f"Missing FMP_API_KEY in environment or .env files. Checked: {searched}")
    return api_key


def configure_openbb_credentials() -> None:
    """Push resolved env credentials into OpenBB before provider calls."""
    load_shared_env()
    try:
        from openbb import obb
    except ImportError:
        return

    fmp_key = resolve_fmp_api_key()
    if fmp_key:
        obb.user.credentials.fmp_api_key = fmp_key

    tiingo_token = str(os.environ.get("TIINGO_TOKEN") or "").strip()
    if tiingo_token:
        if hasattr(obb.user.credentials, "tiingo_token"):
            obb.user.credentials.tiingo_token = tiingo_token

    intrinio_key = str(os.environ.get("INTRINIO_API_KEY") or "").strip()
    if intrinio_key and hasattr(obb.user.credentials, "intrinio_api_key"):
        obb.user.credentials.intrinio_api_key = intrinio_key


@lru_cache(maxsize=1)
def resolve_thetadata_api_key(*, required: bool = False) -> str:
    load_shared_env()
    api_key = str(os.environ.get("THETADATA_API_KEY") or "").strip()
    if required and not api_key:
        searched = ", ".join(str(path) for path in _dotenv_candidates())
        raise ValueError(
            f"Missing THETADATA_API_KEY in environment or .env files. Checked: {searched}"
        )
    return api_key
