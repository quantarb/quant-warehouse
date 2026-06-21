from __future__ import annotations

import threading
import time

from quant_warehouse.refresh.parallel import run_symbol_workers


def test_run_symbol_workers_sequential_preserves_order_of_completion():
    seen: list[str] = []

    def worker(symbol: str) -> list[dict[str, object]]:
        seen.append(symbol)
        return [{"symbol": symbol, "status": "updated"}]

    results = run_symbol_workers(["A", "B", "C"], worker, max_workers=1)
    assert seen == ["A", "B", "C"]
    assert {row["symbol"] for row in results} == {"A", "B", "C"}


def test_run_symbol_workers_runs_concurrently():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def worker(symbol: str) -> list[dict[str, object]]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return [{"symbol": symbol}]

    results = run_symbol_workers(["A", "B", "C", "D"], worker, max_workers=4)
    assert len(results) == 4
    assert max_active > 1