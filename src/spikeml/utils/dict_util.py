
def dic2str(d, null=False, sep=','):
    """
    Convert a dictionary into a string of `key=value` pairs.

    Parameters
    ----------
    d : dict
        Dictionary to format as string.
    null : bool, optional
        If False, skips keys with `None` values. Default is False.
    sep : str, optional
        Separator between key-value pairs. Default is ','.

    Returns
    -------
    str
        Formatted string of key-value pairs.
    """
        
    ss = []
    for key,value in d.items():
        if value is not None:
            ss.append(f'{key}={value}') 
    return sep.join(ss)
               