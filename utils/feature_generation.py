import pandas as pd
import numpy as np
import time
import sys
import datetime as dt
from typing import List, Union
import multiprocessing as mp
import pandas_ta as ta

def _report_progress(job_num, num_jobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + ' done after ' + \
        str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')


def _get_indicators_list() -> List:
    # return indicator_list
    indicators = ['aberration', 'accbands', 'ad', 'adosc', 'adx', 'alma', 'amat',
        'ao', 'aobv', 'apo', 'aroon', 'atr', 'bbands' 'bias', 'bop', 'brar',
        'cci', 'cdl_doji', 'cdl_inside', 'cfo', 'cg', 'chop', 'cksp', 'cmf', 'cmo', 'coppock',
        'decay', 'decreasing', 'dema', 'donchian', 'dpo', 'ebsw', 'efi', 'ema', 'entropy',
        'eom', 'er', 'eri', 'fisher', 'fwma', 'hilo', 'hl2', 'hlc3', 'hma', 'hwc', 'hwma', 'increasing',
        'inertia', 'kama', 'kc', 'kdj', 'kst', 'kurtosis', 'linreg', 'log_return', 'long_run', 'macd', 'mad',
        'massi', 'mcgd', 'median', 'mfi', 'midpoint', 'mom', 'natr', 'nvi', 'obv', 'ohlc4', 'pdist',
        'percent_return', 'pgo', 'ppo', 'psar', 'psl', 'pvi', 'pvo', 'pvol', 'pvr', 'pvt', 'pwma', 'qqe',
        'qstick', 'quantile', 'rma', 'roc', 'rsi', 'rsx', 'rvgi', 'rvi', 'short_run', 'sinwma', 'skew',
        'slope', 'sma', 'smi', 'squeeze', 'ssf', 'stdev', 'stoch', 'stochrsi', 'supertrend', 'swma', 't3',
        'tema', 'thermo', 'trend_return', 'trima', 'trix', 'true_range', 'tsi', 'ttm_trend', 'ui', 'uo',
        'variance', 'vidya', 'vortex', 'vp', 'vwap', 'vwma', 'wcp', 'willr', 'wma', 'zlma', 'zscore']

    # indicators = bars.ta.indicators(as_list=True, exclude=['midprice', 'midpoint', 'ichimoku', 'entropy'])
    return indicators

def _expand_call(kargs):
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def _calc_indicators(bars:pd.DataFrame, indicator: str) -> Union[pd.DataFrame, pd.Series]:
    return bars.ta(kind=indicator)
    


def _calc_features_mp(bars:pd.DataFrame, indicator_list:List, n_jobs:int, verbose=False) -> List:
    results = []

    jobs = [{'func': _calc_indicators,'bars': bars,
            'indicator': indicator} for indicator in indicator_list]
    print(f"Numbers of indicators added to multiprocessing jobs: {len(jobs)}")

    pool = mp.Pool(processes=n_jobs)
    outputs = pool.imap_unordered(_expand_call, jobs)
    
    time0 = time.time()

    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs):
        results.append(out_)
        if verbose:
            task = jobs[i]['indicator'][1]
            _report_progress(i+1, len(jobs), time0, task)

    pool.close()
    pool.join()
    return results


def _create_features_df(outs: List) -> pd.DataFrame:
    df0 = pd.DataFrame(dtype=np.float64)
    for out in outs:
        if out is not None:
            df0 = pd.concat([df0, out], axis=1)

    df1 = df0.sort_index().copy()
    del df0
    return df1


