from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _summary(data: object) -> dict[str, object]:
    shape = getattr(data, "shape", None)
    columns = getattr(data, "columns", None)
    index = getattr(data, "index", None)
    return {
        "type": type(data).__name__,
        "shape": list(shape) if shape is not None else None,
        "columns": [str(value) for value in columns] if columns is not None else None,
        "index_min": str(index.min()) if index is not None and len(index) else None,
        "index_max": str(index.max()) if index is not None and len(index) else None,
    }


def _target_uri(endpoint: str, bucket: str, prefix: str) -> str:
    return f"s3s://{endpoint}:{bucket}?path_prefix={prefix.strip('/')}&aws_auth=true"


def bootstrap(*, source: Path, endpoint: str, bucket: str, prefix: str, log: Path) -> None:
    from quant_warehouse.ingest.credentials import load_shared_env

    load_shared_env()
    from arcticdb import Arctic

    roots = [("shared", source)]
    providers = source / "providers"
    if providers.is_dir():
        roots.extend((f"provider:{path.name}", path) for path in sorted(providers.iterdir()) if path.is_dir())

    log.parent.mkdir(parents=True, exist_ok=True)
    for root_name, root in roots:
        source_store = Arctic(f"lmdb://{root}")
        target_prefix = prefix if root_name == "shared" else f"{prefix.rstrip('/')}/providers/{root_name.split(':', 1)[1]}"
        target_store = Arctic(_target_uri(endpoint, bucket, target_prefix))
        target_libraries = set(target_store.list_libraries())
        for library_name in source_store.list_libraries():
            source_library = source_store.get_library(library_name)
            if library_name not in target_libraries:
                target_store.create_library(library_name)
                target_libraries.add(library_name)
            target_library = target_store.get_library(library_name)
            target_symbols = set(str(value) for value in target_library.list_symbols())
            for symbol in source_library.list_symbols():
                symbol = str(symbol)
                source_item = source_library.read(symbol)
                expected = _summary(source_item.data)
                status = "copied"
                if symbol in target_symbols:
                    actual = _summary(target_library.read(symbol).data)
                    if actual == expected:
                        status = "already_verified"
                    else:
                        target_library.write(symbol, source_item.data, metadata=source_item.metadata, prune_previous_versions=True)
                        status = "replaced_mismatch"
                else:
                    target_library.write(symbol, source_item.data, metadata=source_item.metadata, prune_previous_versions=True)
                actual = _summary(target_library.read(symbol).data)
                if actual != expected:
                    raise RuntimeError(f"verification failed for {root_name}/{library_name}/{symbol}: {expected} != {actual}")
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "root": root_name,
                    "library": str(library_name),
                    "symbol": symbol,
                    "status": status,
                    "summary": expected,
                }
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
                print(f"{status}: {root_name}/{library_name}/{symbol}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap local ArcticDB data into an S3 ArcticDB warehouse.")
    parser.add_argument("--source", type=Path, default=Path("~/.quant-warehouse/arctic").expanduser())
    parser.add_argument("--endpoint", default="s3.us-west-1.amazonaws.com")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="quant-warehouse")
    parser.add_argument("--log", type=Path, default=Path("~/.quant-warehouse/logs/bootstrap-s3.jsonl").expanduser())
    args = parser.parse_args()
    bootstrap(source=args.source, endpoint=args.endpoint, bucket=args.bucket, prefix=args.prefix, log=args.log)


if __name__ == "__main__":
    main()
