import numpy as np
from scipy import stats
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



def htest_connections(MM, ij1, ij2, name=None, normal=False, alpha=0.05):
    """
    Hypothesis test for connection values.
    Automatically method and run the appropriate hypothesis test.
    
    Parameters
    ----------
    MM : np.array (tensor) [runs, i, j])
    ij1: tuple (i,j) : matrix (row,column) in group1
    ij2: tuple (i,j) : matrix (row,column) in group2
    paired : bool, default=False
        Whether samples are paired (e.g., before/after).
    normal : bool, default=True
        Whether data is approximately normally distributed.
        If False, a nonparametric test is used.
    alpha : float, default=0.05
        Significance level for hypothesis decision.
        
    Returns
    -------
    dict
        Dictionary with:
            'name': name of layer
            'test': name of test
            'stat': test statistic
            'p': p-value
            'reject': whether null hypothesis is rejected
    """    
    
    i1,j1 = ij1
    i2,j2 = ij2
    c1 = MM[:,i1,j1]
    c2 = MM[:,i2,j2]

    if normal:
        stat, p = stats.ttest_ind(c1, c2, equal_var=False)
        test = "Independent t-test (Welch)"
    else:
        stat, p = stats.mannwhitneyu(c1, c2)
        test = "Mann-Whitney U test"

    reject = p < alpha
            
    print(f'{name}: [{i1},{j1}] x [{i2},{j2}]]')          
    print('  c1:', c1)
    print('  c2:', c2)
    print("  test:", test)
    print("  statistic:", stat)
    print("  p-value:", p)
    print("  reject_H0:", reject)
    

    return {
        "name": name,
        "test": test,
        "stat": stat,
        "p": p,
        "reject": reject
    }
    return t_stat, p_value

def htest_connector_identity(results, alpha=0.05):
    conns = results.get_connector_tensors(as_map=True)
    for i, (name, MM) in  enumerate(conns.items()):
        shape = MM[0].shape
        for j in range(0, shape[1]):
            for i in range(0, shape[0]):
                if i==j:
                    continue
                htest_connections(MM, (j,j), (i,j), name, alpha=0.05)
