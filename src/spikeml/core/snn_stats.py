import numpy as np
from typing import Any, List, Optional, Union, Tuple

def connector_stats(
    results: list,
    title: Optional[str] = None,
    _type: Optional[str] = None,
    ax: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation of connection matrices across multiple runs.

    This function aggregates results containing connection weight matrices (`M`)
    and computes their element-wise mean and standard deviation. Each element in
    `results` is expected to either be:
      - a NumPy array representing a connection matrix, or
      - an object with an attribute `M` containing the matrix.

    Parameters
    ----------
    results : list of ndarray or list of objects
        Collection of results to aggregate. Each element must be a NumPy array or
        have an attribute `M` containing the connection matrix.
    title : str, optional
        Title for visualization or reporting. Currently unused.
    _type : str, optional
        Optional type label or identifier for downstream processing. Currently unused.

    Returns
    -------
    mean_matrix : ndarray
        Element-wise mean of all connection matrices.
    std_matrix : ndarray
        Element-wise standard deviation of all connection matrices.

    Notes
    -----
    - The function assumes all matrices in `results` have identical shapes.
    - Unused parameters (`title`, `_type`, `callback`, `ax`) are included for
      future extensibility or compatibility with plotting pipelines.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[2, 3], [4, 5]])
    >>> m, sd = connector_stats([a, b])
    >>> m
    array([[1.5, 2.5],
           [3.5, 4.5]])
    >>> sd
    array([[0.5, 0.5],
           [0.5, 0.5]])
    """

    def _get_M(result):
        if isinstance(result, np.ndarray):
            M = result
        else:
            M = result.M
        return M        

    m = None
    for n, result in enumerate(results):
        M = _get_M(result)
        m = M.copy() if m is None else m+M
    m /= len(results) 
    sd2 = np.zeros(m.shape)
    for n, result in enumerate(results):
        M =  _get_M(result)
        sd2 += (m-M)**2
    sd2 /= len(results)
    sd = np.sqrt(sd2)

    return m, sd
