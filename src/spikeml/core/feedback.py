import numpy as np
import math
from typing import Optional, Union
from enum import Enum, auto

from spikeml.core.params import Params, NNParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

def compute_error(
    s: np.ndarray,
    y: np.ndarray,
    params: Optional['SSNNParams'] = None,
    mean: bool = True,
    strict: bool = False
) -> float:
    """
    Compute an element-wise error between a target signal `s` and a predicted
    output `y`, optionally returning the mean error.

    This error function measures mismatches between the desired signal and the
    model output using asymmetric penalties. Let ``vmax`` denote the maximum
    allowable signal value (taken from `params`):

    Non-strict mode (default)
    -------------------------
    The error penalizes false positives only:
        error_i = (vmax - s_i) * y_i

    Strict mode
    -----------
    The error penalizes both false positives and false negatives:
        error_i = s_i * (vmax - y_i) + (vmax - s_i) * y_i

    In binary cases (s, y ∈ {0, vmax}), this reduces to:

        strict = False:
            s  y  err
            0  0   0
            0  1   1
            1  0   0
            1  1   0

        strict = True:
            s  y  err
            0  0   0
            0  1   1
            1  0   1
            1  1   0

    Parameters
    ----------
    s : np.ndarray
        Target signal. May be binary or continuous in the range [vmin, vmax].
    y : np.ndarray
        Predicted signal. Must have the same shape as `s`.
    params : SSNNParams, optional
        Parameters providing `vmax` and `vmin`. If None, a default instance is used.
    mean : bool, optional
        If True (default), return the mean error value. If False, return the
        full error array.
    strict : bool, optional
        Whether to use the strict variant of the error function.

    Returns
    -------
    float or np.ndarray
        Mean error (if ``mean=True``) or element-wise error array otherwise.

    Examples
    --------
    >>> compute_error(np.array([1, 0]), np.array([1, 1]), strict=False)
    0.5

    >>> compute_error(np.array([1, 0]), np.array([1, 1]), strict=True)
    1.0
    """

    
    if params is None:
        params = SSNNParams()
    if strict:
        p = s * (params.vmax-y)+(params.vmax-s)*y
    else:
        p = (params.vmax-s)*y
    err = p.mean(axis=-1) if mean else p
    #print(f's: {s} ; y: {y}', f'-> err: {err:.2f}')
    return err


class OutputAggregation(Enum):
    DP = auto()
    MEAN = auto()
    SUM = auto()
    SUM_CLIP = auto()
    MAX = auto()
    
def xcompute_error(
    s: np.ndarray,
    y: np.ndarray,
    R: int = 1,
    method: str = 'sum+clip',
    mean: bool = True,
    params: Optional['SSNNParams'] = None,
    strict: bool = False
) -> float:
    """
    Compute an error between `s` and `y` with optional temporal downsampling
    or aggregation prior to evaluation. The final error is computed using
    `compute_error()`.

    Downsampling/Aggregation
    ------------------------
    When `R > 1`, the predicted signal `y` is grouped in blocks of length `R`
    and aggregated using the method specified by `method`:

    - 'dp':
        Repeat (duplicate) each value in `s` to match the length of `y`.
    - 'mean':
        y_block = mean of each block of size R.
    - 'sum':
        y_block = sum over blocks, then clipped into [vmin, vmax].
    - 'sum+clip' (default):
        Same as 'sum'.
    - 'max':
        y_block = max value in each block.

    After aggregation, the function calls:

        compute_error(s, y_aggregated, strict=strict)

    Parameters
    ----------
    s : np.ndarray
        Target signal. If method='dp' and lengths mismatch, it is repeated.
    y : np.ndarray
        Predicted signal.
    R : int, optional
        Downsampling factor (block size). Defaults to 1 (no aggregation).
    method : {'dp', 'mean', 'sum', 'sum+clip', 'max'}, optional
        Aggregation method for `y`. Defaults to 'sum+clip'.
    params : SSNNParams, optional
        Provides `vmin` and `vmax`. Defaults to None.
    strict : bool, optional
        Passed through to `compute_error()`.

    Returns
    -------
    float
        Aggregated error between `s` and (optionally downsampled) `y`.

    Examples
    --------
    >>> s = np.array([0, 1])
    >>> y = np.array([0, 1, 1, 1])
    >>> xcompute_error(s, y, R=2, method='mean')
    compute_error([0, 1], [0.5, 1.0])

    >>> xcompute_error(s, y, R=2, method='max')
    compute_error([0, 1], [1, 1])
    """
    
    if params is None:
        params = SSNNParams()
    if R>1:
        # --- Aggregation methods ---
        if method == OutputAggregation.DP:
            # Downsample prediction by duplicating s
            if s.shape[0] != y.shape[0]:
                s = np.repeat(s, R)
        else:
            # reshape y into blocks
            y = y.reshape(y.shape[0] // R, R)
            if method == OutputAggregation.MEAN:
                y = y.mean(axis=1)
            elif method == OutputAggregation.SUM:
                y = y.sum(axis=1)
            elif method == OutputAggregation.SUM_CLIP:
                y = y.sum(axis=1)
                y = np.clip(y, params.vmin, params.vmax)
            elif method == OutputAggregation.MAX:
                y = y.max(axis=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            
    err = compute_error(s, y, params=params, mean=mean, strict=strict)
    return err

def compute_sg(err, params):
    """Compute error based gain.

    Args:
        err (float): the error_
        params (_type_): parameters

    Returns:
        float: gain
    """
    sg = np.exp(-err*params.e_err)
    return sg
