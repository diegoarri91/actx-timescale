import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


def float_to_int(df, columns, int_na=0):
    if isinstance(int_na, int):
        int_na = [int_na] * len(columns)
    df = df[columns].copy()
    for ii, col in enumerate(columns):
        df[col] = df[col].astype(float).round()
        df.loc[df[col].isna(), col] = int_na[ii]
        df[col] = df[col].astype(int)
    return df


def raw_autocorrelation(x, biased=True, negative_times=False):
    raw_autocorr = raw_correlation(x, x, biased=biased)
    if not negative_times:
        raw_autocorr = raw_autocorr[raw_autocorr.shape[0] // 2:]
    return raw_autocorr


def raw_correlation(x1, x2, biased=True):
    if biased:
        n = min(x1.shape[0], x2.shape[0])
    else:
        n = np.arange(1, min(x1.shape[0], x2.shape[0]) + 1, 1)
        n = np.concatenate((n, n[:-1][::-1]))
        n = n.reshape((len(n),) + tuple([1] * (x1.ndim - 1)))

    raw_corr = fftconvolve(x1, x2[::-1], mode='full', axes=0)[::-1] / n

    return raw_corr


def band_matrix(unbanded_matrix, max_band=None, fill_with_nan=False):
    N = unbanded_matrix.shape[1]
    max_band = max_band if max_band is not None else N
    banded_matrix = np.zeros((max_band, N))
    if fill_with_nan:
        banded_matrix = banded_matrix * np.nan
    for diag in range(max_band):
        indices = diag_indices(N, k=diag)
        banded_matrix[diag, :N - diag] = unbanded_matrix[indices]
    return banded_matrix


def diag_indices(n, k=0):
    rows, cols = np.diag_indices(n)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def get_dt(t):
    arg_dt = 20 if len(t) >= 20 else len(t)
    dt = np.mean(np.diff(t[:arg_dt]))
    return dt


def unband_matrix(banded_matrix, symmetric=True, lower=True):
    N = banded_matrix.shape[1]
    unbanded_matrix = np.zeros((N, N))
    for diag in range(banded_matrix.shape[0]):
        indices = diag_indices(N, k=diag)
        unbanded_matrix[indices] = banded_matrix[diag, :N - diag]
    if symmetric:
        indices = np.tril_indices(N)
        unbanded_matrix[indices] = unbanded_matrix.T[indices]
    if not (symmetric) and lower:
        unbanded_matrix = unbanded_matrix.T
    return unbanded_matrix


def list_columns_as_rows(df, list_columns=None, drop_index=False, add_idx=False):
    df = df.reset_index(drop=drop_index)
    order = list(df.columns)

    if list_columns is None:
        scalar_columns = []
        list_columns = []
        for col in df.columns:
            if isinstance(df.iloc[0][col], list) or isinstance(df.iloc[0][col], np.ndarray):
                list_columns += [col]
            else:
                scalar_columns += [col]
    else:
        scalar_columns = [col for col in df.columns if col not in list_columns]

    df_expanded = pd.DataFrame(
        {col: np.repeat(df[col].values, df[list_columns[0]].str.len()) for col in scalar_columns})

    for col in list_columns:
        try:
            col_vals = np.concatenate((df[col].values))
            if col_vals.ndim == 2:
                col_vals = [col_vals[i, ...] for i in range(col_vals.shape[0])]
        except ValueError:
            col_vals = [ii for row in df[col].values for ii in row]
        df_expanded = df_expanded.assign(**{col: col_vals})

    df_expanded = df_expanded[df.columns]

    if add_idx:
        df_expanded['idx'] = np.concatenate([np.arange(0, len(df.iloc[0][col]), 1)] * len(df))

    return df_expanded
