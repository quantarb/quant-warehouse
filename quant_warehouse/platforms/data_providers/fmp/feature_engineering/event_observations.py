"""Known sparse observations; disclosure time, not transaction time, gates inputs."""
from datetime import datetime
import warnings
import polars as pl
from .broad_observations import dated

CHANNELS=['signal_value',*[f'text_{i}' for i in range(7)]]


def event_frame(frame,symbol,family,date_column,values,*,event_column=None):
    if frame.is_empty():return pl.DataFrame()
    frame=dated(frame,column=date_column)
    expressions=[expr.cast(pl.Float32).alias(channel) for channel,expr in zip(CHANNELS,values)]
    expressions += [pl.lit(None,dtype=pl.Float32).alias(c) for c in CHANNELS[len(values):]]
    event=pl.col(event_column).cast(pl.Datetime('ns'),strict=False) if event_column else pl.col('date')
    return frame.select(pl.lit(symbol).alias('symbol'),'date',event.alias('event_date'),pl.lit(family).alias('target_family'),*expressions)


def issuer_event_observations(warehouse,symbol,*,start='1900-01-01',end='2026-09-09'):
    for section in ('ownership_government_trades','ownership_insider_trading','estimates_price_target','dividends','historical_splits'):
        frame=warehouse.read_fundamentals(symbol,section=section,start=start,end=end)
        if frame.is_empty():continue
        if section=='ownership_government_trades':
            disclosure=next((c for c in ('disclosure_date','date') if c in frame.columns),None)
            # Some stored panels contain only dated, all-missing placeholders.
            # They are not government-trade observations and have no disclosure date.
            frame=frame.filter(pl.any_horizontal([
                pl.col(c).cast(pl.String).str.to_lowercase().is_in(['nan','']).not_().fill_null(False)
                for c in ('transaction_type','amount')
            ]))
            if frame.is_empty():continue
            if disclosure is None:
                warnings.warn(
                    f'{symbol}: skipping {frame.height} government-trade observations without a disclosure date',
                    RuntimeWarning, stacklevel=2,
                )
                continue
            kind=pl.col('transaction_type').cast(pl.String).str.to_lowercase()
            amounts=pl.col('amount').cast(pl.String).str.replace_all(r'[$,]','').str.extract_all(r'\d+').list.eval(pl.element().cast(pl.Float64))
            values=[kind.str.contains('purchase').cast(pl.Float32),kind.str.contains('sale').cast(pl.Float32),amounts.list.first(),amounts.list.last()]
            # OpenBB FMP maps disclosureDate/dateReceived to date; older snapshots retain disclosure_date.
            out=event_frame(frame,symbol,'equity.ownership.government_trades',disclosure,values,event_column='transaction_date')
        elif section=='ownership_insider_trading':
            kind=pl.col('transaction_type').cast(pl.String).str.to_uppercase()
            values=[kind.str.starts_with('P').cast(pl.Float32),kind.str.starts_with('S').cast(pl.Float32),
                    pl.col('securities_owned'),pl.col('securities_transacted'),pl.col('transaction_price'),
                    (pl.col('acquisition_or_disposition').cast(pl.String)=='A').cast(pl.Float32),
                    (pl.col('acquisition_or_disposition').cast(pl.String)=='D').cast(pl.Float32)]
            out=event_frame(frame,symbol,'equity.ownership.insider_trading','filing_date',values,event_column='transaction_date')
        elif section=='estimates_price_target':
            out=event_frame(frame,symbol,'equity.estimates.price_target','published_date',
                            [pl.col(c) for c in ['price_target','adj_price_target','price_when_posted']])
        elif section=='dividends':
            # Ex-date observations are known by EOD even where announcement history is missing.
            out=event_frame(frame,symbol,'equity.calendar.dividend','ex_dividend_date',[pl.col('amount').cast(pl.String),pl.col('adjusted_amount')])
        else:
            out=event_frame(frame,symbol,'equity.calendar.splits','date',[pl.col('numerator'),pl.col('denominator')])
        yield section,out.filter(pl.col('date').is_between(datetime.fromisoformat(start),datetime.fromisoformat(end)))
