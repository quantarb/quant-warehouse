from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

ProgressLogger = Callable[[str], None] | None
SymbolWorker = Callable[[str], list[dict[str, object]]]


def run_symbol_workers(
    symbols: Sequence[str],
    worker: SymbolWorker,
    *,
    max_workers: int = 1,
    progress_logger: ProgressLogger = None,
    progress_label: str = "refresh",
) -> list[dict[str, object]]:
    """Run a per-symbol worker sequentially or across a thread pool."""
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    total = len(normalized)
    if total == 0:
        return []

    if max(1, int(max_workers)) <= 1:
        results: list[dict[str, object]] = []
        for index, symbol in enumerate(normalized, start=1):
            results.extend(worker(symbol))
            if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
                progress_logger(f"Warehouse {progress_label} progress: {index:,}/{total:,} symbols processed")
        return results

    results: list[dict[str, object]] = []
    completed = 0
    progress_lock = threading.Lock()

    def _run(symbol: str) -> None:
        nonlocal completed
        try:
            rows = worker(symbol)
        except Exception as exc:
            rows = [
                {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ]
        with progress_lock:
            results.extend(rows)
            completed += 1
            if callable(progress_logger) and (completed == 1 or completed % 25 == 0 or completed == total):
                progress_logger(
                    f"Warehouse {progress_label} progress: {completed:,}/{total:,} symbols processed"
                )

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(_run, symbol) for symbol in normalized]
        for future in as_completed(futures):
            future.result()
    return results
