"""Bounded Polars feature-family observations for multi-rate consumers.

Keep individual fields. No family averaging, future filling, or corpus-fitted
statistics. The caller persists one issuer's observations before reading another.
"""
from collections import Counter
from hashlib import blake2b
import re
import polars as pl

META={'symbol','cik','fiscal_period','fiscal_year','calendar_year','period','reported_currency'}
DATE_FIELDS=('filing_date','accepted_date','published_date','observation_date','as_of','date','period_ending','__index__')


def dated(frame, *, column=None):
    if frame.is_empty():return pl.DataFrame(schema={'date':pl.Datetime('ns')})
    candidates=[column] if column else [c for c in DATE_FIELDS if c in frame.columns and
        (frame.schema[c].is_temporal() or frame.schema[c]==pl.String)]
    if not candidates:raise ValueError('Source has no usable observation date')
    expressions=[]
    for name in candidates:
        value=pl.col(name)
        if frame.schema[name]==pl.String:value=value.str.to_datetime(strict=False)
        elif isinstance(frame.schema[name],pl.Datetime) and frame.schema[name].time_zone:
            value=value.dt.convert_time_zone('UTC').dt.replace_time_zone(None)
        expressions.append(value.cast(pl.Datetime('ns')).dt.truncate('1d'))
    return frame.with_columns(pl.coalesce(expressions).alias('date')).drop_nulls('date').sort('date')



def numeric(frame):
    return [c for c,t in frame.schema.items() if c not in META and c not in DATE_FIELDS and t.is_numeric()
            and frame[c].cast(pl.Float64).is_finite().any()]


def select_fields(frame,fields):
    return frame.select('date',*[pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64)).alias(c) for c in fields]).group_by('date').agg(pl.all().last()).sort('date')


def ratio_family(column, *, metrics=False):
    rules=(('valuation',('ev_to','market_cap','enterprise_value','graham','invested_capital')),
           ('profitability',('return_on','income_quality','roic')),
           ('solvency',('debt','coverage','working_capital'))) if metrics else (
           ('profitability',('margin','return_on','income_quality','tax_burden','interest_burden','sga_to_revenue')),
           ('liquidity',('current_ratio','quick_ratio','cash_ratio','operating_cash_flow_ratio','working_capital')),
           ('solvency',('debt','coverage','equity_multiplier','liabilities','solvency')),
           ('efficiency',('turnover','days_','cycle','cash_conversion')),
           ('valuation',('price_to','ev_to','market_cap','enterprise_value','yield','book_value_per_share','earnings_per_share','revenue_per_share','capex_per_share')))
    return next(('ft_ratios_'+family for family,tokens in rules if any(t in column.lower() for t in tokens)),None)


def safe_divide(a,b):
    value=a/b
    return pl.when((b!=0)&value.is_finite()).then(value)


def news_features(frame,buckets=32):
    frame=dated(frame,column='published_at')
    texts=[c for c in ('title','excerpt','body') if c in frame.columns]
    if not texts or frame.is_empty():return pl.DataFrame()
    rows=[]
    for row in frame.select('date',*texts).iter_rows(named=True):
        counts=Counter(int.from_bytes(blake2b(t.encode(),digest_size=8).digest(),'little')%buckets
                       for t in re.findall(r'[a-z][a-z0-9]{2,}',' '.join(str(row[c] or '') for c in texts).lower()))
        if counts:rows.append({'date':row['date'],**{f'text_bucket_{i:02d}':float(counts[i]) for i in range(buckets)}})
    if not rows:return pl.DataFrame()
    return pl.DataFrame(rows).group_by('date').agg(pl.all().sum()).sort('date')


def build_issuer_families(warehouse,symbol,*,start='1900-01-01',end='2026-09-09'):
    """Yield (family, rate, frame, provenance) without a fused universe panel."""
    def read(section,period=None):
        return dated(warehouse.read_fundamentals(symbol,section=section,period=period,start=start,end=end))
    market=read('historical_market_cap')
    if 'market_cap' not in market.columns:raise ValueError(f'{symbol}: missing historical market cap')
    market=select_fields(market,['market_cap'])
    for period,rate,suffix in [('annual','annual','annual'),('quarter','quarterly','quarterly')]:
        for section in ('income','balance','cash'):
            frame=read(section,period);fields=numeric(frame)
            if not fields:continue
            aligned=select_fields(frame,fields).join_asof(market.rename({'market_cap':'_mcap'}),on='date',strategy='backward')
            values=aligned.select('date',*[safe_divide(pl.col(c),pl.col('_mcap')).alias(c) for c in fields])
            yield f'fmp_{section}_mcap_{suffix}',rate,values,{'section':section,'period':period,'denominator':'historical market cap on or before observation'}
        grouped={}
        for section in ('ratios','metrics'):
            raw=read(section,period)
            if raw.is_empty():
                continue
            if 'fiscal_period' not in raw.columns:raise ValueError(f'{symbol}/{section}/{period}: missing fiscal period')
            # TTM snapshot rows from older refreshes do not belong to these families.
            expected=['FY'] if period=='annual' else ['Q1','Q2','Q3','Q4']
            raw=raw.filter(pl.col('fiscal_period').cast(pl.String).is_in(expected))
            for c in numeric(raw):
                family=ratio_family(c,metrics=section=='metrics')
                if family:grouped.setdefault(family,[]).append(select_fields(raw,[c]).rename({c:f'{section}_{c}'}))
        for family,parts in grouped.items():
            merged=parts[0]
            for part in parts[1:]:merged=merged.join(part,on='date',how='full',coalesce=True)
            yield family+'_'+suffix,rate,merged.sort('date'),{'sections':['ratios','metrics'],'period':period,'exclude_ttm':True}
        for section,kind in [('income_growth','income'),('balance_growth','balance'),('cash_growth','cash')]:
            frame=read(section,period);fields=[c for c in numeric(frame) if c.startswith('growth_')]
            if fields:yield f'ft_growth_{kind}_{suffix}',rate,select_fields(frame,fields),{'section':section,'period':period}
    # Daily valuation families use only backward as-of joins to observed statements.
    aligned=market
    for section in ('income','balance','cash'):
        raw=read(section,'quarter');fields=numeric(raw)
        if fields:aligned=aligned.join_asof(select_fields(raw,fields).rename({c:f'{section}__{c}' for c in fields}),on='date',strategy='backward')
    available=set(aligned.columns)
    if {'balance__total_debt','balance__cash_and_cash_equivalents'}<=available:
        aligned=aligned.with_columns((pl.col('market_cap')+pl.col('balance__total_debt')-pl.col('balance__cash_and_cash_equivalents')).alias('_ev'))
    for denominator,prefix in [('market_cap','mcap'),('_ev','ev')]:
        if denominator not in aligned.columns:continue
        fields=[c for c in aligned.columns if c.startswith(('income__','cash__','balance__'))]
        for inverse,kind in [(False,'yield'),(True,'multiple')]:
            expressions=[]
            for c in fields:
                a,b=pl.col(c),pl.col(denominator)
                expressions.append(safe_divide(b,a).alias(c) if inverse else safe_divide(a,b).alias(c))
            yield f'fmp_daily_{prefix}_{kind}','daily',aligned.select('date',*expressions),{'denominator':denominator,'join':'backward as-of','period':'quarter'}
    for section,family,period in [('employee_count','fmp_employee_count',None),('esg_score','fmp_esg_scores',None),
            ('ratings_historical','fmp_historical_ratings',None),('ownership_institutional','fmp_institutional_position_summary',None),
            ('estimates_historical','fmp_quarterly_financial_estimates_quarterly','quarter')]:
        frame=read(section,period);fields=numeric(frame)
        if fields:yield family,'daily',select_fields(frame,fields),{'section':section,'period':period,'dates':'recorded source observations'}
    from datetime import datetime
    raw=warehouse.read_news(symbol,provider='fmp',start=datetime.fromisoformat(start),end=datetime.fromisoformat(end))
    news=news_features(raw)
    if not news.is_empty():yield 'fmp_company_news','daily',news,{'section':'company_news','encoding':'32 deterministic token-count buckets, no fitted vocabulary','date':'publication'}


def group_contexts(panel,groups):
    """Historical peer aggregates from a caller-specified universe and group map."""
    joined=panel.join(groups.lazy(),on='symbol',how='inner').sort('symbol','date')
    joined=joined.with_columns(*[(pl.col('close')/pl.col('close').shift(h).over('symbol')-1).alias(f'return_{h}') for h in (1,5,20,60,120,252)])
    metrics=[c for c in panel.collect_schema().names() if c.startswith('pe_')]
    for group in ('sector','industry'):
        usable=joined.filter(pl.col(group).is_not_null() & (pl.col(group)!=''))
        yield group+'_performance',group,usable.group_by('date',group).agg(pl.col(f'return_{h}').mean() for h in (1,5,20,60,120,252)).collect(engine='streaming')
        yield group+'_pe',group,usable.group_by('date',group).agg(pl.col(c).median() for c in metrics).collect(engine='streaming')


def calendar_features(dates):
    return dates.select('date').unique().with_columns(
        pl.col('date').dt.weekday().cast(pl.Float32).alias('weekday'),
        pl.col('date').dt.month().cast(pl.Float32).alias('month'),
        pl.col('date').dt.ordinal_day().cast(pl.Float32).alias('day_of_year'))
