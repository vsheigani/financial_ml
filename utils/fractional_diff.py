from statsmodels.tsa.stattools import adfuller
import numpy as np
from pandas import Series
from numba import njit

def calc_min_d(feature_series: Series, col_name: 'str', thresh: np.float64):
    min_d = None

    for d_value in np.linspace(0, 1, 11):
        # Applying fractional differentiation
        diffed_series = fast_frac_diff(feature_series, diff_amt=d_value, col_name=col_name, thresh=thresh).dropna()
        if(diffed_series.shape[0] < 100):
            continue
        adf_result = adfuller(diffed_series, maxlag=1, regression='c', autolag=None)

        if adf_result[0] < adf_result[4]['5%']:
            if d_value != 0.0:
                min_d = round(d_value + 0.1, 1)
                return min_d
    return min_d


@njit(parallel=True)
def fast_get_weights(diff_amt, thresh, limit):
    weights = np.ones((1, 1), dtype=np.float64)
    k = np.float64(1.0)
    ctr = np.float64(0.0)
    while True:
        # compute the next weight
        weight_ = -np.float64(weights[0][0]) * np.float64(diff_amt - k + 1.0) / k
        weight_ = np.array([[weight_]])
        if abs(weight_[0][0]) <= thresh:
            break
        weights = np.vstack((weight_, weights))
        k += 1
        ctr += 1
        if ctr == limit - 1: 
            break
    return weights


@njit(parallel=True)
def run_frac_diff(feat: np.array, df_index: np.array, weights: np.array):
    width = len(weights) - 1
    feat_size = len(feat)
    temp_df_ = np.zeros(len(df_index), dtype=np.float64)
    for iloc1 in range(width, feat_size):
        # loc0 = feat_index[iloc1 - width]
        # loc1 = df_index[iloc1]
        temp_df_[iloc1] = np.dot(weights.T, feat[iloc1-width:iloc1+1].ravel())[0]
    return temp_df_


def fast_frac_diff(feat_series, col_name, diff_amt: int = 0.4, thresh=1e-5):
    # 1) Compute weights for the longest series
    df_index = feat_series.index.to_numpy(dtype=np.int64)
    # 2.2) compute fractionally differenced series for each stock
    weights = fast_get_weights(diff_amt, thresh, feat_series.shape[0])

    feat = feat_series.ffill().dropna()
    feat = feat.to_numpy(dtype=np.float64)
    
    frac_df_ = run_frac_diff(feat, df_index, weights)
    output_series = Series(frac_df_, index=feat_series.index, name=col_name)

    for i in range(0, len(weights)-1):
        if output_series.iloc[i] == 0.0:
            output_series.iloc[i] = np.nan
        else:
            break
    return output_series
