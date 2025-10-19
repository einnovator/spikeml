import numpy as np
import math
from typing import Optional, Union

from spikeml.core.params import Params, NNParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

def compute_error(
    s: np.ndarray,
    y: np.ndarray,
    normalize: bool = False,
    params: Optional['SSNNParams'] = None
) -> float:
    """
    Compute the mean error between target signal `s` and predicted output `y`.

    The error is defined as:
        p = s * (vmax - y) + (vmax - s) * y
    and averaged over all elements.

    Parameters
    ----------
    s : np.ndarray
        Target signal (binary or continuous between vmin and vmax).
    y : np.ndarray
        Predicted signal (same shape as `s`).
    normalize : bool, optional
        Placeholder for future normalization (currently unused). Default is False.
    params : SSNNParams, optional
        Network parameters, must provide `vmax` and `vmin`. If None, a default SSNNParams is used.


    s y err
    0 0 0 
    0 1 1
    1 0 1
    1 1 0

    Returns
    -------
    float
        Mean error between `s` and `y`.
    """
    
    if params is None:
        params = SSNNParams()
    p = s * (params.vmax-y)+(params.vmax-s)*y
    err = p.mean()
    #print(f's: {s} ; y: {y}', f'-> err: {err:.2f}')
    return err

def xcompute_error(
    s: np.ndarray,
    y: np.ndarray,
    R: int = 1,
    method: str = 'sum+clip',
    params: Optional['SSNNParams'] = None
) -> float:
    """
    Compute the error between `s` and `y` with optional downsampling or aggregation.

    Downsampling/aggregation is performed by reshaping `y` and applying one of the following methods:
    - 'dp': repeat `s` to match `y`
    - 'mean': average over blocks of length `R`
    - 'sum': sum over blocks of length `R` and clip to [vmin, vmax]
    - 'sum+clip': sum and clip (default)
    - 'max': maximum over blocks of length `R`

    Parameters
    ----------
    s : np.ndarray
        Target signal.
    y : np.ndarray
        Predicted signal.
    R : int, optional
        Downsampling factor or block size. Default is 1 (no downsampling).
    method : str, optional
        Aggregation method. Default is 'sum+clip'.
    params : SSNNParams, optional
        Network parameters providing `vmin` and `vmax`. Defaults to None.

    Returns
    -------
    float
        Mean error after aggregation.
    """
    if params is None:
        params = SSNNParams()
    if R>1:
        if method=='dp':
            if s.shape[0]!=y.shape[0]:
                s = np.repeat(s, R)
        elif method=='mean':
            y = y.reshape(y.shape[0] // R, R).mean(axis=1)
        elif method=='sum':        
            y = y.reshape(y.shape[0] // R, R).sum(axis=1)
            y =  np.clip(y, params.vmin, params.vmax)
        elif method=='sum+clip':        
            y = y.reshape(y.shape[0] // R, R).sum(axis=1)
            y =  np.clip(y, params.vmin, params.vmax)
        elif method=='max':        
            y = y.reshape(y.shape[0] // R, R).max(axis=1)
    p = s * (params.vmax-y)+(params.vmax-s)*y
    err = p.mean()
    #print(f's: {s} ; y: {y}', f'-> err: {err:.2f}')
    return err

def compute_sg(err, params):
    sg = math.exp(-err*params.e_err)
    return sg
