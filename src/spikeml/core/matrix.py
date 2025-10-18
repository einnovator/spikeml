import numpy as np
from typing import Optional, Tuple, Union, Any

def matrix_split(M):
    """
    Split a matrix into its positive and negative components.

    Parameters
    ----------
    M : np.ndarray
        Input matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple `(Mp, Mn)` where:
        - `Mp` contains only the positive elements of `M` (negative values set to 0)
        - `Mn` contains only the negative elements of `M` (positive values set to 0)
    """
    
    Mp,Mn = M.copy(),M.copy()
    Mp[M < 0] = 0
    Mn[M > 0] = 0
    return Mp, Mn

def normalize_matrix(
    M: np.ndarray,
    c_in: Union[int, float, bool, None] = 1,
    c_out: Union[int, float, bool, None] = 0,
    abs_: bool = True,
    strict: bool = False,
    debug: bool = False
) -> np.ndarray:
    """
    Normalize a connection matrix by its input and/or output sums.

    Each row or column is scaled to ensure the sum of absolute values
    does not exceed the corresponding normalization constant.

    Parameters
    ----------
    M : np.ndarray
        Matrix to normalize.
    c_in : int | float | bool | None, default=1
        Normalization factor applied across rows (input dimension).
        If `True`, uses the number of columns.
    c_out : int | float | bool | None, default=0
        Normalization factor applied across columns (output dimension).
        If `True`, uses the number of rows.
    abs_ : bool, default=True
        If True, normalization is based on the absolute values of elements.
    strict : bool, default=False
        If True, strictly enforces normalization factors without threshold smoothing.
    debug : bool, default=False
        If True, prints diagnostic matrices during normalization.

    Returns
    -------
    np.ndarray
        Normalized matrix.

    Notes
    -----
    - The normalization is applied separately to the positive and negative
      components of the matrix.
    - If both `c_in` and `c_out` are `None`, the matrix is returned unchanged.
    """
    if isinstance(c_in, bool):
        c_in = M.shape[1]
    if isinstance(c_out, bool):
        c_out = M.shape[0]
    Mp,Mn = matrix_split(M)

    def _normalize_matrix(M):
        if c_in is not None and c_in>0:
            M_ = np.abs(M) if abs_ else M
            Mw = M_.sum(axis=1, keepdims=True)
            if not strict:
                b = (Mw>c_in).astype(int)
                Mw[Mw < c_in] = 1
                Mw[Mw >= c_in] *= 1/c_in
            M = M / Mw
            if debug:
                print(f'Mrows:\n{Mw}')
        if c_out is not None and c_out>0:
            M_ = np.abs(M) if abs_ else M
            Mw = M_.sum(axis=0, keepdims=True)
            if not strict:
                b = (Mw>c_out).astype(int)
                Mw[Mw < c_out] = 1
                Mw[Mw >= c_out] *= 1/c_out
            M = M / Mw
            if debug:
                print(f'Mcols:\n{Mw}')
        return M

    Mp = _normalize_matrix(Mp)
    Mn = _normalize_matrix(Mn)
    M = Mp+Mn
    return M


def _mult(a, b):
    """
    Elementwise multiply two matrices if both are defined.

    Parameters
    ----------
    a, b : np.ndarray or None
        Input arrays. If one of them is None, the other is returned.

    Returns
    -------
    np.ndarray or None
        Elementwise product if both exist, otherwise whichever is not None.
    """    
    return a*b if a is not None and b is not None else a if a is not None else b 
   
def cmask(M, c_in, c_out, abs_=True):
    """
    Compute input/output masks for matrix connectivity constraints.

    Parameters
    ----------
    M : np.ndarray
        Matrix to mask.
    c_in : float or None
        Maximum allowed sum per input (row). If None, ignored.
    c_out : float or None
        Maximum allowed sum per output (column). If None, ignored.
    abs_ : bool, default=True
        If True, masking is based on the absolute values of `M`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        `(d, d_in, d_out)` where:
        - `d` is the combined mask (d_in * d_out)
        - `d_in` masks rows that exceed `c_in`
        - `d_out` masks columns that exceed `c_out`
    """    
    M_ = np.abs(M) if abs_ else M
    d_out,d_in = None,None
    if c_out is not None and c_out>0:
        s_out=M_.sum(axis=0)
        d_out = np.zeros(M.shape)
        d_out[:,s_out >= c_out] = 0
        d_out[:,s_out < c_out] = 1
    else:
        d_out = np.ones(M.shape)            
    if c_in is not None and c_in>0:
        s_in=M_.sum(axis=1)
        d_in = np.zeros(M.shape)
        d_in[s_in >= c_in,:] = 0
        d_in[s_in < c_in,:] = 1
    else:
        d_in = np.ones(M.shape)
    #print('s_out:', s_out, 's_in:', s_in)
    d = _mult(d_in,d_out)
    return d,d_in,d_out 

def cmask2(
    M: np.ndarray,
    c_in: Optional[float],
    c_out: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute separate positive/negative connectivity masks and merge them.

    Parameters
    ----------
    M : np.ndarray
        Input matrix.
    c_in : float or None
        Input normalization limit.
    c_out : float or None
        Output normalization limit.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Combined `(d, d_in, d_out)` masks after merging positive and negative components.
    """
    Mp,Mn = matrix_split(M)
    d_p,d_in_p,d_out_p = cmask(Mp,c_in, c_out)
    d_n,d_in_n,d_out_n = cmask(Mn, c_in, c_out)
    d,d_in,d_out = _mult(d_p, d_n),_mult(d_in_p, d_in_n),_mult(d_out_p, d_out_n)
    return d,d_in,d_out


def matrix_init(
    params: Optional[Any] = None,
    size: Optional[Union[int, Tuple[int, int], list[int]]] = None
) -> Optional[np.ndarray]:
    """
    Initialize a random or constant connection matrix based on parameters.

    Parameters
    ----------
    params : ConnectorParams, optional
        Connection parameter object defining mean, standard deviation,
        normalization constants, etc.
    size : int or tuple of int, optional
        Matrix shape `(rows, cols)`. If an integer is provided, creates a square matrix.

    Returns
    -------
    np.ndarray or None
        Initialized matrix, or None if insufficient information is provided.

    Notes
    -----
    - If `params.sd == 0`, the matrix is filled with the mean value.
    - If `params.c_in` or `params.c_out` are specified, normalization is applied.
    """
    if params is None:
        params = ConnectorParams()
    if size is None:
        size = params.size
    if size is None:
        return None
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        size = [size, size]
    if params.sd==0:
        M = np.zeros(size)   
        if params.mean!=0:
            M.fill(params.mean)
    else:
        M = np.random.normal(loc=params.mean, scale=params.sd, size=size)
    if (params.c_in is not None and params.c_in>0) or (params.c_out is not None and params.c_out>0):
        M = normalize_matrix(M, c_in=params.c_in, c_out=params.c_out)
    return M

def matrix_init2(
    params: Any,
    params2: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize two matrices representing positive and negative connections.

    Parameters
    ----------
    params : ConnectorParams
        Parameters for generating the positive matrix.
    params2 : ConnectorParams, optional
        Parameters for generating the negative matrix. If None, uses `params`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `(Mp, Mn)` where:
        - `Mp` is the positive (absolute) matrix.
        - `Mn` is the negative (absolute) matrix.
    """
    Mp = np.abs(matrix_init(params))
    if params2 is None:
        params2 = params
    Mn = -np.abs(matrix_init(params2))
    return Mp,Mn

