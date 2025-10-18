import numpy as np
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