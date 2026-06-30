from __future__ import annotations

import platform


def _linux_arm64() -> bool:
    return platform.system() == "Linux" and platform.machine().lower() in {"aarch64", "arm64"}


def require_arcticdb() -> None:
    """Raise a helpful error if ArcticDB is missing."""
    try:
        import arcticdb  # noqa: F401
    except ImportError as exc:
        if _linux_arm64():
            raise ImportError(
                "ArcticDB is required. PyPI has no Linux arm64 wheel, so pip cannot "
                "install it automatically. Install into this env first:\n"
                "  conda install -c conda-forge arcticdb\n"
                "Then reinstall quant-warehouse if needed:\n"
                "  pip install -e /path/to/quant-warehouse"
            ) from exc
        raise ImportError(
            "ArcticDB is required. It should install with quant-warehouse on this "
            "platform. Try:\n"
            "  pip install 'quant-warehouse'"
        ) from exc
