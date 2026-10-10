import os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from filelock import Timeout
import pytest
from quant_warehouse.platforms.data_providers.thetadata.write_coordination import option_store_write_lock

def test_same_lock_is_reentrant_and_independent_writer_blocked(tmp_path):
    backend=SimpleNamespace(_uri='lmdb://'+str(tmp_path/'store'))
    code="from types import SimpleNamespace;from quant_warehouse.platforms.data_providers.thetadata.write_coordination import option_store_write_lock;import sys\nwith option_store_write_lock(SimpleNamespace(_uri=sys.argv[1]),timeout=0):pass"
    with option_store_write_lock(backend):
        with option_store_write_lock(backend):pass
        result=subprocess.run([sys.executable,'-c',code,backend._uri],capture_output=True)
        assert result.returncode and b'Timeout' in result.stderr
    assert subprocess.run([sys.executable,'-c',code,backend._uri],capture_output=True).returncode==0

def test_unknown_or_remote_store_is_not_claimed_serialized():
    with pytest.raises(ValueError):option_store_write_lock(object())
    with pytest.raises(ValueError):option_store_write_lock(SimpleNamespace(_uri='s3://example'))

def test_provider_override_uses_actual_provider_root(tmp_path):
    backend=SimpleNamespace(config=SimpleNamespace(provider_arctic_uri=lambda p:'lmdb://'+str(tmp_path/p)))
    with option_store_write_lock(backend):assert (tmp_path/'thetadata/.option-chain-rmw.lock').exists()

def test_concurrent_refreshes_preserve_both_updates_and_prior_version(tmp_path):
    import polars as pl
    from quant_warehouse.platforms.data_providers.thetadata import options as opt
    from quant_warehouse.warehouse.backend import ArcticBackend
    from quant_warehouse.warehouse.storage import provider_library
    uri='lmdb://'+str(tmp_path/'arctic');backend=ArcticBackend(uri)
    initial=opt.normalize_thetadata_option_chain(pl.DataFrame({'symbol':['AAPL'],'expiration':['2026-11-20'],'strike':[100.],'right':['CALL'],'created':['2026-10-09 16:00:00-04:00'],'bid':[1.],'ask':[2.]}))
    opt.write_option_chain_arctic('AAPL',initial,backend=backend)
    code='''import sys,time
from pathlib import Path
import polars as pl
from quant_warehouse.platforms.data_providers.thetadata import options as opt
from quant_warehouse.warehouse.backend import ArcticBackend
backend=ArcticBackend(sys.argv[1]);ready=Path(sys.argv[2]);ready.touch()
while not Path(sys.argv[3]).exists():time.sleep(.01)
original=opt.read_provider_frame
def delayed(*a,**k):
 frame=original(*a,**k);time.sleep(.3);return frame
opt.read_provider_frame=delayed
frame=opt.normalize_thetadata_option_chain(pl.DataFrame({'symbol':['AAPL'],'expiration':['2026-11-20'],'strike':[float(sys.argv[4])],'right':['CALL'],'created':['2026-10-09 16:00:00-04:00'],'bid':[3.],'ask':[4.]}))
opt.write_option_chain_arctic('AAPL',frame,backend=backend)
'''
    start=tmp_path/'start';jobs=[]
    try:
        for strike in (110,120):jobs.append(subprocess.Popen([sys.executable,'-c',code,uri,str(tmp_path/str(strike)),str(start),str(strike)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
        deadline=time.monotonic()+15
        while not all((tmp_path/str(s)).exists() for s in (110,120)):
            assert time.monotonic()<deadline;time.sleep(.01)
        start.touch()
        for job in jobs:
            _,error=job.communicate(timeout=15);assert job.returncode==0,error.decode()
    finally:
        for job in jobs:
            if job.poll() is None:job.kill();job.wait()
    # Parent cached version0 before both other writers. Its subsequent write must
    # refresh that cache under the lock, preserving their updates rather than
    # merging against its old snapshot.
    own=initial.with_columns(pl.lit(130.).alias('strike'),pl.lit('AAPL261120C00130000').alias('contract_symbol'))
    opt.write_option_chain_arctic('AAPL',own,backend=backend)
    latest=opt.read_option_chain_arctic('AAPL',backend=backend)
    assert sorted(latest['strike'].to_list())==[100.,110.,120.,130.]
    library=backend._library(provider_library(opt.OPTIONS_THETADATA_EOD_LIBRARY,opt.OPTIONS_THETADATA_PROVIDER))
    assert len(library.list_versions('AAPL'))==4
    assert library.read('AAPL',as_of=0).data['strike'].tolist()==[100.]
