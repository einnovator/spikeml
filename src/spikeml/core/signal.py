import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import random
import math
import copy
from typing import Annotated, Any, Callable, Optional, Sequence, Union, Tuple, Dict, List
from pydantic import BaseModel, Field, WithJsonSchema
import pydantic
import ipywidgets as widgets

def signal_dc(dim=2, T=10, s=None, value=1):
    """
    Generate a constant (DC) signal over time.

    Creates a signal consisting of `T` repeated copies of a fixed vector `s`.
    If `s` is not provided, a zero vector of dimension `dim` is used.
    If `s` is an integer, it is converted into a one-hot vector.

    Parameters
    ----------
    dim : int, optional
        Dimension of the signal vector. Default is 2.
    T : int, optional
        Length of the signal in time steps. Default is 10.
    s : int | ndarray | None, optional
        Source value for the signal:
        - If `None`, a zero vector is used.
        - If an integer, a one-hot vector is generated.
        - If a NumPy array, it is used directly.
        Default is `None`.
    value : float, optional
        Value assigned to the active element when generating a one-hot vector.
        Default is 1.

    Returns
    -------
    ss : ndarray of shape (T, dim)
        The generated constant signal repeated for `T` time steps.
    """
    if isinstance(s,np.ndarray):
        pass
    elif s is None or s<0:
        s = np.zeros(dim)
    else:
        s = encode1_onehot(s, dim, value=value)
    if len(s.shape)==1:
        s = np.expand_dims(s, axis=0)
    ss = np.repeat(s, T, axis=0)
    return ss

def signal_pulse(dim=2, T=10, L=1, s=[], value=1):
    """
    Generate a pulsed signal sequence composed of repeated constant segments.

    Each symbol in `s` generates a constant (DC) segment of duration `T`.
    The full pattern is repeated `L` times.

    Parameters
    ----------
    dim : int, optional
        Dimension of each signal vector. Default is 2.
    T : int, optional
        Duration (in time steps) of each pulse segment. Default is 10.
    L : int, optional
        Number of repetitions of the full pulse sequence. Default is 1.
    s : list of int or list of ndarray, optional
        List of symbols or vectors defining each pulse segment.
        Each element is converted into a one-hot vector if it is an integer.
        Default is an empty list.
    value : float, optional
        Value assigned to the active element in one-hot encoding.
        Default is 1.

    Returns
    -------
    ndarray
        Concatenated sequence of all generated pulses with shape (L * len(s) * T, dim).
    """

    a = []
    for l in range(0, L):
        for s_ in s:
            a.append(signal_dc(dim, T, s_, value=value))
    return np.concatenate(a)

def encode1_onehot(sym, dim, value=1.0):
    """
    Encode a single symbol as a one-hot vector.

    Parameters
    ----------
    sym : int
        Symbol index to encode (0 ≤ sym < dim).
    dim : int
        Size of the one-hot vector.
    value : float, optional
        Value assigned to the active element. Default is 1.0.

    Returns
    -------
    ndarray of shape (dim,)
        One-hot encoded vector.
    """

    a = np.zeros(dim, dtype=float)
    a[sym] = value
    return a

def encode_onehot(ss, nsym):
    """
    Encode an array of integer symbols into one-hot vectors.

    Parameters
    ----------
    ss : ndarray of int
        Input array of symbols to encode.
        Can be 1D (shape: [N]) or 2D (shape: [T, N]).
    nsym : int
        Number of possible symbols (size of the one-hot dimension).

    Returns
    -------
    ndarray
        One-hot encoded representation of shape:
        - (N, nsym) if input is 1D
        - (T, N, nsym) if input is 2D
    """

    if len(ss.shape)==2:
        a = np.zeros([ss.shape[0], ss.shape[1], nsym])
        for i in range(0, ss.shape[0]):
            for j in range(0, ss.shape[1]):
                e = np.zeros([nsym], dtype=float)
                e[ss[i,j]] = 1.0
                a[i,j,:] = e    
    else:
        a = np.zeros([ss.shape[0], nsym])
        for j in range(0, ss.shape[0]):
            e = np.zeros([nsym], dtype=float)
            e[ss[j]] = 1.0
            a[j,:] = e            
    return a

def signal_changes(data):
    """
    Detect time indices where the signal changes.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Time series data of shape (T, dim), or a list of such arrays.

    Returns
    -------
    ndarray of int
        Indices (time steps) where the signal changes value compared to the previous step.
    """
    
    if type(data)==list: data = np.stack(data)
    dv = np.any(data[1:] != data[:-1], axis=1)
    x = np.where(dv)[0] + 1
    return x

def signal_unique(data, E=0):
    """
    Extract unique signal vectors within a given tolerance.

    Parameters
    ----------
    data : ndarray of shape (N, dim)
        Input signal vectors.
    E : float, optional
        Tolerance for considering two vectors equivalent.
        Default is 0 (exact match).

    Returns
    -------
    ndarray of shape (K, dim)
        Unique vectors from the input, filtered by the tolerance threshold.
    """

    unique = []
    for a in data:
        if not any(np.sum(np.abs(a - u)) <= E for u in unique):
            unique.append(a)
    return np.array(unique)

def signal_ranges(data, ref, E=0):
    """
    Compute contiguous index ranges where the signal matches reference patterns.

    Parameters
    ----------
    data : ndarray of shape (T, dim)
        Input time series signal.
    ref : ndarray of shape (R, dim)
        Reference signal patterns to search for in `data`.
    E : float, optional
        Tolerance threshold for vector equality. Default is 0.

    Returns
    -------
    list[list[tuple[int, int]]]
        A list of lists, where each sublist corresponds to a reference pattern,
        containing tuples (start_idx, end_idx) for contiguous matching ranges.
    """

    ranges = []
    for i, x in enumerate(ref):
        cc = np.sum(np.abs(data - x), axis=-1) <= E
        #print(cc)
        ii = np.argwhere(cc).flatten()
        #print(ii)
        diffs = np.diff(ii)
        ii_ = np.where(diffs != 1)[0]
        idx,idx0 = ii[ii_], ii[ii_+1]
        #print(idx, idx0)
        l = []
        i0 = ii[0]
        for i,i1 in enumerate(idx):
            l.append((i0,i1))
            i0 = idx0[i]
        l.append((i0,ii[-1]))
        ranges.append(l)
    return ranges


def stats_per_input(
    data: Union[np.ndarray, list[np.ndarray]],
    signal: np.ndarray,
    f: Callable[[np.ndarray], Any],
    E: float = 0,
    aggregate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics of `data` grouped by unique signal values.

    This function groups samples based on unique input signal vectors (within a
    numerical tolerance `E`) and applies a user-defined function `f` to compute
    aggregate statistics (e.g., mean, variance) over each group.  
    It can also optionally compute per-range statistics instead of aggregated ones.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Data values corresponding to each signal observation.
        Must have the same number of samples (first dimension) as `signal`.
        If a list is provided, it will be stacked along a new axis.
    signal : ndarray of shape (T, D)
        Input signal vectors corresponding to each data sample.
        Each row represents a D-dimensional signal at a given time step.
    f : callable
        Aggregation function applied to each subset of `data`
        corresponding to a unique signal (or signal range).
        Examples include `np.mean`, `np.std`, or a user-defined function.
    E : float, optional
        Numerical tolerance used to treat two signals as identical.
        If `E = 0` (default), grouping is based on exact equality.
    aggregate : bool, optional
        If True (default), computes a single aggregated value for each unique signal.
        If False, computes statistics over sequential ranges of signal repetitions
        using `signal_ranges`.

    Returns
    -------
    ref : ndarray of shape (K, D)
        Unique or representative signal vectors.
        Each row corresponds to one signal group.
    - When `aggregate=True`: 
        size : ndarray of shape (K,), optional
            Number of samples in each signal group.
            Returned only when `aggregate=True`.
        values : ndarray, ndarray of shape (K, ...) with one value per group.
            Results of applying `f` to grouped data: ndarray of shape (K, ...) with one value per group.
    - When `aggregate=False`: list of lists, where each sublist contains
        ranges : list of tuple, Range for each representative signal vector:
        values : list, Results of applying `f` to grouped data:
            list of lists, where each sublist contains
             per-range statistics for each unique signal.

    Notes
    -----
    - This function is commonly used in analyzing spiking neural networks,
      simulations, or time-series experiments where repeated signal inputs
      produce multiple response samples.
    - The grouping tolerance `E` allows robust comparison of floating-point signals.
    - When `aggregate=False`, `signal_ranges()` is expected to provide
      contiguous index ranges for repeated signal segments.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> signal = np.array([[0], [0], [1], [1]])
    >>> stats_per_input(data, signal, np.mean)
    (array([[0.],
            [1.]]),
     array([2, 2]),
     array([1.5, 3.5]))

    Example with non-aggregated ranges:
    >>> stats_per_input(data, signal, np.mean, aggregate=False)
    (array([[0.],
            [1.]]),
     [ [(1.5)], [(3.5)] ])
    """


    if type(data)==list:
        data = np.stack(data)
    ref = signal_unique(signal, E=E)
    #print(ref)
    #c = signal_changes(sx_)
    #print(c)
    if aggregate:
        size = []
        values = []
        for i,x in enumerate(ref):
            ii = np.argwhere(np.sum(np.abs(signal - x), axis=-1) <= E).flatten()
            #print(i, type(ii), ii)
            data_ = data[ii]
            #if len(data_.shape)>2:
            #    data_ = data_.reshape(data_.shape[0], -1)
            value = f(data_)
            sz = data_.shape[0]
            size.append(sz)
            values.append(value)
            #print(i, ':', x, sz, f'{mean:.4f}')
            
        size = np.array(size)
        values = np.array(values)
        return ref, size, values
    else:
        ranges = signal_ranges(signal, ref=ref, E=E)
        values = []
        for i,rr in enumerate(ranges):
            values_ = []
            values.append(values_)
            for i0,i1 in rr:
                #print(i, rr)
                data_ = data[i0:(i1+1)]
                value = f(data_)
                values_.append(value)
                #print(i, ':', x, sz, f'{mean:.4f}')
        #size = np.array(size)
        #values = np.array(values)
        return ref, ranges, values 


def mean_per_input(
    data: Union[np.ndarray, list[np.ndarray]],
    signal: np.ndarray,
    E: float = 0,
    aggregate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean data value grouped by unique signal inputs.

    For each unique signal vector (within tolerance `E`), compute the
    average of the corresponding `data` values.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Array of values associated with each time step of the signal.
    signal : ndarray of shape (T, dim)
        Input signal vectors corresponding to each data point.
    E : float, optional
        Tolerance for grouping signal vectors as identical. Default is 0.
    aggregate: bool = True

    Returns
    -------

    ref : ndarray of shape (K, D)
        Unique or representative signal vectors.
        Each row corresponds to one signal group.
    - When `aggregate=True`: 
        size : ndarray of shape (K,), optional
            Number of samples in each signal group.
            Returned only when `aggregate=True`.
        means : ndarray
            Results of applying `f` to grouped data: ndarray of shape (K, ...) with one value per group.
    - When `aggregate=False`: list of lists, where each sublist contains
        ranges : list of tuple, Range for each representative signal vector:
        means : list, ndarray of shape (K, ...) with one mean value per group.
    """

    return stats_per_input(data, signal, np.mean, E=E, aggregate=aggregate)

def var_per_input(
    data: Union[np.ndarray, list[np.ndarray]],
    signal: np.ndarray,
    E: float = 0,
    aggregate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute variance of data value grouped by unique signal inputs.

    For each unique signal vector (within tolerance `E`), compute the
    average of the corresponding `data` values.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Array of values associated with each time step of the signal.
    signal : ndarray of shape (T, dim)
        Input signal vectors corresponding to each data point.
    E : float, optional
        Tolerance for grouping signal vectors as identical. Default is 0.
    aggregate: bool = True


    Returns
    -------
    ref : ndarray of shape (K, D)
        Unique or representative signal vectors.
        Each row corresponds to one signal group.
    - When `aggregate=True`: 
        size : ndarray of shape (K,), optional
            Number of samples in each signal group.
            Returned only when `aggregate=True`.
        var : ndarray
           Variance value of `data` for each unique input.
    - When `aggregate=False`: list of lists, where each sublist contains
        ranges : list of tuple, Range for each representative signal vector:
        var :  list, ndarray of shape (K, ...) with one variance value per group.
    """

    return stats_per_input(data, signal, np.var, E=E, aggregate=aggregate)

def std_per_input(
    data: np.ndarray,
    signal: np.ndarray,
    E: float = 0,
    aggregate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute standard deviation of data value grouped by unique signal inputs.

    For each unique signal vector (within tolerance `E`), compute the
    average of the corresponding `data` values.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Array of values associated with each time step of the signal.
    signal : ndarray of shape (T, dim)
        Input signal vectors corresponding to each data point.
    E : float, optional
        Tolerance for grouping signal vectors as identical. Default is 0.
    aggregate: bool = True

    Returns
    -------
    ref : ndarray of shape (K, D)
        Unique or representative signal vectors.
        Each row corresponds to one signal group.
    - When `aggregate=True`: 
        size : ndarray of shape (K,), optional
            Number of samples in each signal group.
            Returned only when `aggregate=True`.
        stds : ndarray
           Standard deviation of `data` for each unique input.
    - When `aggregate=False`: list of lists, where each sublist contains
        ranges : list of tuple, Range for each representative signal vector:
        stds :  list, ndarray of shape (K, ...) with one Standard deviation value per group.
    """

    return stats_per_input(data, signal, np.std, E=E, aggregate=aggregate)


def sum_per_input(
    data: np.ndarray,
    signal: np.ndarray,
    E: float = 0,
    axis: int = 0,
    aggregate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute standard deviation of data value grouped by unique signal inputs.

    For each unique signal vector (within tolerance `E`), compute the
    average of the corresponding `data` values.

    Parameters
    ----------
    data : list[ndarray] or ndarray
        Array of values associated with each time step of the signal.
    signal : ndarray of shape (T, dim)
        Input signal vectors corresponding to each data point.
    E : float, optional
        Tolerance for grouping signal vectors as identical. Default is 0.
    axis: int = 0
        Axis to sum over
    aggregate: bool = True


   Returns
    -------
    ref : ndarray of shape (K, D)
        Unique or representative signal vectors.
        Each row corresponds to one signal group.
    - When `aggregate=True`: 
        size : ndarray of shape (K,), optional
            Number of samples in each signal group.
            Returned only when `aggregate=True`.
        sums : ndarray
           Sume of `data` for each unique input on axis.
    - When `aggregate=False`: list of lists, where each sublist contains
        ranges : list of tuple, Range for each representative signal vector:
        sums :  list, ndarray of shape (K, ...) with one sum value per group.    """

    return stats_per_input(data, signal, lambda a: np.sum(a, axis=0), E=E, aggregate=aggregate)

