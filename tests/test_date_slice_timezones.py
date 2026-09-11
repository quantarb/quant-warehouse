from datetime import datetime, timezone
import polars as pl
import pytest
from quant_warehouse.warehouse.prices import _slice_dates

@pytest.mark.parametrize('zone',[None,'UTC','America/New_York'])
def test_date_slice_preserves_source_timezone_and_inclusive_bounds(zone):
    frame=pl.DataFrame({'as_of':[datetime(2023,12,31),datetime(2024,1,1),datetime(2024,1,2)]})
    if zone:
        frame=frame.with_columns(pl.col('as_of').dt.replace_time_zone(zone))
    result=_slice_dates(frame,start='2024-01-01',end='2024-01-01',date_column='as_of')
    assert result.height==1
    assert result.schema==frame.schema
    assert result['as_of'][0].day==1

def test_offset_boundaries_compare_instants():
    frame=pl.DataFrame({'date':[datetime(2024,1,1,tzinfo=timezone.utc)]})
    assert _slice_dates(frame,start='2023-12-31T19:00:00-05:00',end='2024-01-01T00:00:00Z').height==1
