import numpy as np
from scipy.signal import convolve
from enum import Enum
from typing import Optional, Tuple, Union, Any

def vec2int(x: Union[np.ndarray, list], vmin: float = 0.0, vmax: float = 1.0) -> int:
    """
    Convert a numerical vector to an integer by thresholding and interpreting it as a binary number.

    Each element of the vector is compared to the midpoint `(vmin + (vmax - vmin)/2)`. 
    Values below the midpoint are set to 0, and values greater than or equal to the midpoint are set to 1.
    The resulting binary vector is then interpreted as a single integer (big-endian).

    Parameters
    ----------
    x : np.ndarray or list
        Input numerical vector (1D array or list of numbers).
    vmin : float, optional
        Minimum value of the range. Default is 0.0.
    vmax : float, optional
        Maximum value of the range. Default is 1.0.

    Returns
    -------
    int
        Integer representation of the thresholded binary vector.

    Example
    -------
    >>> import numpy as np
    >>> vec2int(np.array([0.1, 0.6, 0.8]))
    3  # binary '011' -> decimal 3
    """
    
    v = vmin + (vmax-vmin)/2
    x[x<v]=0
    x[x>=v]=1
    x = x.astype(int)
    return int("".join(x.astype(str)), 2)


def _sum(x: Optional[np.ndarray], y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Safely sum two arrays, allowing for None values.

    Parameters
    ----------
    x : np.ndarray or None
        First array.
    y : np.ndarray or None
        Second array.

    Returns
    -------
    np.ndarray or None
        Sum of x and y if both are not None, otherwise the non-None array, or None if both are None.
    """
    return x+y if x is not None and y is not None else x if x is not None else y


class UpsampleMethod(Enum):
    """
    Enumeration of available methods for upsampling vectors or signals.

    Each member defines a distinct strategy for generating higher-resolution
    representations of an input array when increasing its sampling rate.

    Members
    -------
    ZEROS : int
        Insert zeros between original samples (zero-stuffing).
    REPEAT : int
        Repeat each sample value to fill the upsampled positions.
    NORMAL : int
        Interpolate intermediate values using a Gaussian-weighted distance
        around each original sample.
    NORMAL0 : int
        Non-vector implementation of NORMAL.
        
    Notes
    -----
    - These methods are typically used in signal processing or neural modeling
      to control how temporal or spatial resolution is increased.
    """
    ZEROS = 1
    REPEAT = 2
    NORMAL = 3
    NORMAL0 = 4

def upsample(
    x: np.ndarray,
    R: int = 1,
    method: Optional[UpsampleMethod] = UpsampleMethod.REPEAT,
    sigma: float = 1.0
) -> float:
    """
    Upsample vector (1D array) to higher dimension by a factor R.
    Applying one of the following methods:

    - 'repeat': repeat vector R times

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    R : int, optional
        Upsampling factor (number of interpolated samples per original sample). Default is 1 (no upsampling).
    method : UpsamplingMethod, optional
        Upsampling method: Default is UpsampleMethod.REPEAT.
        - UpsampleMethod.ZEROS   : insert (R-1) zeros between samples
        - UpsampleMethod.REPEAT  : repeat each sample R times
        - UpsampleMethod.NORMAL: smooth normal Gaussian-weighted interpolation
    sigma : float, optional
        Standard deviation for Gaussian weighting (used only if method=UpsampleMethod.NORMAL).

    Returns
    -------
    np.ndarray
        Upsampled vector of length len(x) * R.

    """
    if R < 1:
        raise ValueError("Upsampling factor R must be >= 1")
    if R>1:
        if method is None:
            UpsampleMethod.REPEAT
        if method==UpsampleMethod.REPEAT:
            x = np.repeat(x, R)
        elif method==UpsampleMethod.ZEROS:
            y = np.zeros(len(x) * R, dtype=x.dtype)
            y[::R] = x
            x = y
        elif method==UpsampleMethod.NORMAL:
            # Step 1: zero-insertion upsampling
            y = np.zeros(len(x) * R, dtype=float)
            y[::R] = x
            # Step 2: build Gaussian kernel
            radius = int(3 * sigma * R)  # kernel covers ±3σ
            t = np.arange(-radius, radius + 1) / R
            kernel = np.exp(-0.5 * (t / sigma) ** 2)
            kernel /= kernel.sum()

            # Step 3: smooth via convolution
            x = convolve(y, kernel, mode="same")
        elif method==UpsampleMethod.NORMAL0:  
            N = len(x)
            M = N * R
            y = np.zeros(M, dtype=float)
            # Target (upsampled) indices mapped to original coordinates
            xi = np.arange(M) / R
            for i, pos in enumerate(xi):
                # Compute distance to each original sample
                d = np.arange(N) - pos
                # Gaussian weights based on distance
                w = np.exp(-0.5 * (d / sigma) ** 2)
                w /= w.sum()  # normalize
                y[i] = np.dot(w, x)
            x = y
        else:
            raise ValueError(f"Unknown upsampling method: {method}")
    return x


def upsample_gaussian(x: np.ndarray, R: int, sigma: float = 1.0) -> np.ndarray:
    """
    Upsample a 1D vector by a factor R using .

    Parameters
    ----------
    """
    return upsampled


def downsample(
    y: np.ndarray,
    R: int = 1,
    method: str = 'sum+clip'
) -> float:
    """
    Downsampling/aggregation vector by reshaping `y` and applying one of the following methods:
    - 'mean': average over blocks of length `R`
    - 'sum': sum over blocks of length `R`
    - 'max': maximum over blocks of length `R`

    Parameters
    ----------
    y : np.ndarray
        The vector.
    R : int, optional
        Downsampling factor or block size. Default is 1 (no downsampling).
    method : str, optional
        Aggregation method. Default is 'sum'.
    Returns
    -------
    float
        Mean error after aggregation.
    """
    if R>1:
        if method=='dp':
            if s.shape[0]!=y.shape[0]:
                s = np.repeat(s, R)
        elif method=='mean':
            y = y.reshape(y.shape[0] // R, R).mean(axis=1)
        elif method=='sum':        
            y = y.reshape(y.shape[0] // R, R).sum(axis=1)
        elif method=='max':        
            y = y.reshape(y.shape[0] // R, R).max(axis=1)
    return y
