"""Serialize local option-chain read/merge/write operations across processes.

LMDB serializes individual transactions. A provider read followed by a merge and
write is a larger operation; this lock prevents competing refreshes losing rows.
Readers keep using ArcticDB snapshots and do not acquire this writer lock.
"""
from contextlib import nullcontext
from pathlib import Path
import threading
from filelock import FileLock

OPTIONS_WRITE_PROTOCOL='lmdb-options-rmw-v1'
_LOCKS={}
_REGISTRY_LOCK=threading.Lock()

def option_store_write_lock(backend, *, timeout=30):
    if getattr(backend,'kind',None)=='memory':return nullcontext()
    uri=getattr(backend,'_uri',None)
    if uri is None and hasattr(backend,'config'):
        uri=backend.config.provider_arctic_uri('thetadata')
    if not isinstance(uri,str) or not uri.startswith('lmdb://'):
        raise ValueError('Verified local LMDB store required for option write coordination')
    root=Path(uri.removeprefix('lmdb://')).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    path=str(root/'.option-chain-rmw.lock')
    with _REGISTRY_LOCK:
        lock=_LOCKS.setdefault(path,FileLock(path,timeout=timeout))
    return lock.acquire(timeout=timeout)
