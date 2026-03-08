import numpy as np
import pandas as pd
from .cusum import cusum_filter
from .multiprocessing import mp_pandas_obj
from typing import Union


def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """Find the first profit-taking or stop-loss touch for each event slice."""
    events_ = events.loc[molecule]
    out = events_[['vertical']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['target']
    else:
        profit_taking = pd.Series(index=events.index, dtype=float)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['target']
    else:
        stop_loss = pd.Series(index=events.index, dtype=float)  # NaNs

    out['take_profit'] = pd.Series(dtype=events.index.dtype)
    out['stop_loss'] = pd.Series(dtype=events.index.dtype)

    # Get events
    print(events_['vertical'].head())
    # print(events_['vertical'].head())
    # print(type(events_['vertical'][0]))
    # print(np.float64(close.index[-1]))
    # print(type(np.float64(close.index[-1])))
    for loc, vertical_barrier in events_['vertical'].fillna(close.index[-1]).items():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.at[loc, 'stop_loss'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.at[loc, 'take_profit'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out


def get_vertical_barriers(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """Build vertical barrier timestamps at a fixed time offset from events."""
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))

    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)

    return vertical_barriers


def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times: Union[bool,pd.Series]=False, side_prediction=None, verbose=True):
    """Create triple-barrier events and apply PT/SL with optional side."""

    target = target.reindex(t_events)
    target = target[target > min_ret]

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.DataFrame({'vertical': vertical_barrier_times, 'target': target, 'side': side_}, index=target.index)
    events = events.dropna(subset=['target'])

    # Apply Triple Barrier
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      close=close,
                                      events=events,
                                      pt_sl=pt_sl_,
                                      verbose=verbose)
    for ind in events.index:
        # print(first_touch_dates.loc[ind, :].dropna().min())
        if first_touch_dates.loc[ind, :].dropna().min() is not np.nan:
            # Store the earliest touch as an int64 ns timestamp.
            events.at[ind, 'vertical'] = float(first_touch_dates.loc[ind, :].dropna().min().value)
        else:
            events.at[ind, 'vertical'] = np.nan
    if side_prediction is None:
        events = events.drop('side', axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events['take_profit'] = pt_sl[0]
    events['stop_loss'] = pt_sl[1]

    return events


def get_bins(triple_barrier_events, close):
    """Compute labels/returns for triple-barrier events, with meta labeling."""
    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=['vertical'])
    all_dates = events_.index.union(other=pd.to_datetime(events_['vertical'], unit='ns').array).drop_duplicates()
    prices = close.reindex(all_dates, method='bfill')

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df['return'] = np.log(prices.loc[pd.to_datetime(events_['vertical'], unit='ns').array].array) - np.log(prices.loc[events_.index])
    out_df['target'] = events_['target']

    # Meta labeling: Events that were correct will have pos returns
    if 'side' in events_:
        out_df['return'] = out_df['return'] * events_['side']  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = get_barriers_hit(out_df, triple_barrier_events)

    # Meta labeling: label incorrect events with a 0
    if 'side' in events_:
        out_df.loc[out_df['return'] <= 0, 'bin'] = 0

    # Transform the log returns back to normal returns.
    out_df['return'] = np.exp(out_df['return']) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']

    return out_df



def get_barriers_hit(out_df, events):
    """Assign -1/0/1 labels based on which barrier was touched first."""
    store = []
    for date_time, values in out_df.iterrows():
        ret = values['return']
        target = values['target']

        pt_level_reached = ret > np.log(1 + target) * events.loc[date_time, 'take_profit']
        sl_level_reached = ret < -np.log(1 + target) * events.loc[date_time, 'stop_loss']

        if ret > 0.0 and pt_level_reached:
            # Top barrier hit
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            # Bottom barrier hit
            store.append(-1)
        else:
            # Vertical barrier hit
            store.append(0)

    # Save to 'bin' column and return
    out_df['bin'] = store
    return out_df


def get_daily_volatility(bars, lookback=50):
    """Estimate daily volatility using EWM std of daily returns."""
    new_bars = bars.copy(deep=True)
    new_bars = new_bars[~new_bars.index.duplicated()]
    close = new_bars.close
    df0 = new_bars.index.searchsorted(new_bars.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))

    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=lookback).std()

    sign = pd.Series(np.sign(close-close.shift(1)), index=df0.index)
    sign[sign == 0] = 1
    df0 = df0 * sign
    
    daily_vol = pd.DataFrame(df0, index=bars.index)
    daily_vol = daily_vol.bfill().rename(columns={'close': 'std'})
    daily_vol = pd.Series(daily_vol['std'], index=daily_vol.index)
    return daily_vol


# def get_vertical_barriers(bars, t_events, num_days):
#     """Return vertical barrier timestamps num_days after each event."""
#     t1 = bars.index.searchsorted(t_events+pd.Timedelta(days=num_days))
#     t1 = t1[t1 < bars.shape[0]]
#     t1 = pd.Series(np.int64(bars.index[t1]), index=np.int64(t_events[:t1.shape[0]]), dtype=np.int64) # type: ignore
#     return t1


def get_barrier_events(bars: pd.DataFrame, n_jobs: int,
                       min_ret=0.01, num_days_avg_vol=10,
                       avg_events_per_day=20,
                       pt_sl=[1.0, 1.0],
                       side_prediction: pd.Series|None=None) -> pd.DataFrame:
    """End-to-end pipeline: CUSUM events -> volatility -> triple barrier events."""

    daily_vol = get_daily_volatility(bars, lookback=10)
    cusum_events = cusum_filter(bars['close'],
                                threshold=daily_vol/avg_events_per_day,
                                time_stamps=True)

    vertical_barriers = get_vertical_barriers(t_events=cusum_events, bars=bars, num_days=0, num_hours=5)

    print(f"Volatility averaged for the last {num_days_avg_vol} days")
    print(f"1/{avg_events_per_day} of average daily volatility used as filter_threshold for the cumsum filter")

    triple_barrier_events = get_events(close=bars['close'],
                                        t_events=cusum_events,
                                        pt_sl=pt_sl,
                                        target=daily_vol,
                                        min_ret=min_ret,
                                        num_threads=n_jobs,
                                        vertical_barrier_times=vertical_barriers,
                                        side_prediction=side_prediction)
    return triple_barrier_events
