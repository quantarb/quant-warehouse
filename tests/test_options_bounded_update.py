from datetime import datetime
import polars as pl
import pytest
from quant_warehouse.platforms.data_providers.thetadata import options as opt
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.storage import provider_library

def frame(day,strikes):
 return opt.normalize_thetadata_option_chain(pl.DataFrame({'symbol':['V']*len(strikes),'expiration':['2026-11-20']*len(strikes),'strike':strikes,'right':['CALL']*len(strikes),'created':[day+' 16:00:00-04:00']*len(strikes),'bid':[1.]*len(strikes),'ask':[2.]*len(strikes)}))

def test_bounded_update_preserves_other_days_missing_contracts_metadata_and_versions(tmp_path):
 b=ArcticBackend('lmdb://'+str(tmp_path/'store'))
 opt.write_option_chain_arctic('V',pl.concat([frame('2026-10-08',[100.]),frame('2026-10-09',[100.,110.])]),backend=b)
 lib=b._library(provider_library(opt.OPTIONS_THETADATA_EOD_LIBRARY,'thetadata'))
 metadata=lib.read_metadata('V').metadata
 original=b.read
 def bounded(*a,**k):
  assert k.get('date_range') is not None
  return original(*a,**k)
 b.read=bounded
 opt.write_option_chain_arctic('V',frame('2026-10-09',[100.,120.]).with_columns(pl.lit(3.).alias('bid')),backend=b)
 b.read=original
 result=opt.read_option_chain_arctic('V',backend=b)
 assert result.height==4
 assert result.filter(pl.col('snapshot_date')==datetime(2026,10,8)).height==1
 assert sorted(result.filter(pl.col('snapshot_date')==datetime(2026,10,9))['strike'].to_list())==[100.,110.,120.]
 assert lib.read_metadata('V').metadata==metadata
 assert len(lib.list_versions('V'))==2
 assert len(lib.read('V',as_of=0).data)==3
 # Repeat is idempotent in rows, while prior version remains retained.
 opt.write_option_chain_arctic('V',frame('2026-10-09',[100.,120.]).with_columns(pl.lit(3.).alias('bid')),backend=b)
 assert opt.read_option_chain_arctic('V',backend=b).height==4

def test_new_populated_schema_field_fails_before_write(tmp_path):
 b=ArcticBackend('lmdb://'+str(tmp_path/'store'));opt.write_option_chain_arctic('V',frame('2026-10-09',[100.]),backend=b)
 with pytest.raises(ValueError,match='schema migration'):
  opt.write_option_chain_arctic('V',frame('2026-10-09',[110.]).with_columns(pl.lit(1.).alias('unreviewed_payload')),backend=b)
 assert opt.read_option_chain_arctic('V',backend=b).height==1


def test_legacy_named_timestamp_index_is_preserved(tmp_path):
 b=ArcticBackend('lmdb://'+str(tmp_path/'store'))
 library=provider_library(opt.OPTIONS_THETADATA_EOD_LIBRARY,'thetadata')
 lib=b._library(library)
 initial=opt._prepare_option_chain_for_arctic(frame('2026-10-08',[100.]))
 data=initial.to_pandas().set_index('date');data.index.name='timestamp'
 lib.write('V',data,metadata={'source':'legacy'},prune_previous_versions=False)
 opt.write_option_chain_arctic('V',frame('2026-10-09',[110.]),backend=b)
 assert lib.get_description('V').index[0].name=='timestamp'
 assert len(lib.read('V').data)==2
 assert lib.read_metadata('V').metadata=={'source':'legacy'}
 assert len(lib.list_versions('V'))==2
