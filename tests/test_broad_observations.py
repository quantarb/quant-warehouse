from datetime import datetime
import polars as pl
import pytest
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broad_observations import dated,news_features,safe_divide
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.event_observations import issuer_event_observations


def test_sparse_filing_metadata_does_not_drop_recorded_history():
    frame=pl.DataFrame({'period_ending':[datetime(2000,1,1),datetime(2001,1,1)],'filing_date':[None,datetime(2001,2,1)]})
    assert dated(frame)['date'].to_list()==[datetime(2000,1,1),datetime(2001,2,1)]


def test_news_preserves_individual_buckets_and_publication_dates():
    frame=pl.DataFrame({'published_at':[datetime(2024,1,1)],'title':['profit profit loss']})
    result=news_features(frame)
    assert result.width==33
    assert sum(result.drop('date').row(0))==3


def test_government_event_is_not_available_until_disclosed():
    class Warehouse:
        def read_fundamentals(self,symbol,*,section,**kwargs):
            if section!='ownership_government_trades':return pl.DataFrame()
            return pl.DataFrame({'transaction_date':[datetime(2014,2,10)],'disclosure_date':['2014-02-27'],
                                 'transaction_type':['Sale (Partial)'],'amount':['$15,001 - $50,000']})
    _,frame=next(issuer_event_observations(Warehouse(),'A'))
    assert frame['date'][0]==datetime(2014,2,27)
    assert frame['event_date'][0]==datetime(2014,2,10)
    assert frame['signal_value'][0]==0 and frame['text_0'][0]==1
    assert frame['text_1'][0]==15001 and frame['text_2'][0]==50000


def test_zero_denominators_stay_missing():
    frame=pl.DataFrame({'a':[1.,1.],'b':[0.,2.]})
    assert frame.select(safe_divide(pl.col('a'),pl.col('b')))[:,0].to_list()==[None,.5]


def test_event_normalization_and_merge_preserve_same_day_analysts():
    from quant_warehouse.ingest.normalize import normalize_panel_frame
    from quant_warehouse.warehouse.merge import merge_panel_upsert
    frame=pl.DataFrame({'published_date':[datetime(2024,1,1)]*2,'analyst_name':['Alice','Bob'],
                        'analyst_firm':['Firm A','Firm B'],'price_target':[100.,120.],'news_title':['Raises target','Cuts target']})
    normalized=normalize_panel_frame(frame,provider='fmp')
    assert normalized.height==2
    assert normalized['analyst_name'].to_list()==['Alice','Bob']
    merged=merge_panel_upsert(normalized,normalized)
    assert merged.height==2


def test_unknown_insider_direction_remains_missing_with_numeric_null_columns():
    class Warehouse:
        def read_fundamentals(self, symbol, *, section, **kwargs):
            if section != 'ownership_insider_trading':
                return pl.DataFrame()
            return pl.DataFrame({
                'filing_date': [datetime(2023, 1, 5)],
                'transaction_date': [datetime(2023, 1, 3)],
                'transaction_type': pl.Series([None], dtype=pl.Float64),
                'acquisition_or_disposition': pl.Series([None], dtype=pl.Float64),
                'securities_owned': [100.], 'securities_transacted': [10.],
                'transaction_price': [20.],
            })
    _, frame = next(issuer_event_observations(Warehouse(), 'X'))
    assert frame['date'][0] == datetime(2023, 1, 5)
    assert frame['event_date'][0] == datetime(2023, 1, 3)
    for column in ['signal_value', 'text_0', 'text_4', 'text_5']:
        assert frame[column][0] is None


def test_absent_optional_ratio_history_does_not_drop_issuer_features():
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broad_observations import build_issuer_families
    class Warehouse:
        def read_fundamentals(self, symbol, *, section, **kwargs):
            if section == 'historical_market_cap':
                return pl.DataFrame({'date': [datetime(2023, 1, 1)], 'market_cap': [100.]})
            if section == 'income':
                return pl.DataFrame({'period_ending': [datetime(2023, 1, 1)], 'revenue': [20.]})
            return pl.DataFrame()
        def read_news(self, *args, **kwargs):
            return pl.DataFrame()
    families = dict((name, frame) for name, _, frame, _ in build_issuer_families(Warehouse(), 'X'))
    assert families['fmp_income_mcap_annual']['revenue'].to_list() == [.2]
    assert not any(name.startswith('ft_ratios_') for name in families)


def test_government_all_missing_placeholders_are_not_events():
    class Warehouse:
        def read_fundamentals(self, symbol, *, section, **kwargs):
            if section != 'ownership_government_trades':
                return pl.DataFrame()
            return pl.DataFrame({'transaction_date': [datetime(2017,9,8)],
                                 'transaction_type': [float('nan')], 'amount': [None]})
    assert list(issuer_event_observations(Warehouse(), 'ASML')) == []


def test_government_missing_disclosure_does_not_use_transaction_date_or_stop_other_events():
    class Warehouse:
        def read_fundamentals(self, symbol, *, section, **kwargs):
            if section == 'ownership_government_trades':
                return pl.DataFrame({'transaction_date': [datetime(2024, 1, 1)],
                                     'transaction_type': ['Purchase'], 'amount': ['$1,001 - $15,000']})
            if section == 'historical_splits':
                return pl.DataFrame({'date': [datetime(2024, 2, 1)], 'numerator': [2.], 'denominator': [1.]})
            return pl.DataFrame()
    with pytest.warns(RuntimeWarning, match='A: skipping 1 government-trade observations without a disclosure date'):
        observations = list(issuer_event_observations(Warehouse(), 'A'))
    assert len(observations) == 1
    assert observations[0][0] == 'historical_splits'


def test_government_openbb_date_preserves_disclosure_time():
    class Warehouse:
        def read_fundamentals(self, symbol, *, section, **kwargs):
            if section != 'ownership_government_trades':
                return pl.DataFrame()
            return pl.DataFrame({'date': [datetime(2024, 2, 1)], 'transaction_date': [datetime(2024, 1, 1)],
                                 'transaction_type': ['Purchase'], 'amount': ['$1,001 - $15,000']})
    _, frame = next(issuer_event_observations(Warehouse(), 'A'))
    assert frame['date'][0] == datetime(2024, 2, 1)
    assert frame['event_date'][0] == datetime(2024, 1, 1)


def test_explicit_missing_date_column_reports_source_schema():
    with pytest.raises(ValueError, match="Source has no observation date column 'date'"):
        dated(pl.DataFrame({'transaction_date': [datetime(2024, 1, 1)]}), column='date')
