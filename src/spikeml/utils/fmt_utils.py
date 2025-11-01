
import numpy as np

def fmt_floats(xx, d=4):
    fmt = f"{{:.{d}f}}"
    s = ', '.join([fmt.format(x) for x in xx])
    return f'[{s}]' 
         
def fmt_int(xx):
    if isinstance(xx, np.ndarray):
        xx = xx.tolist()
        return fmt_int(xx)
    if isinstance(xx, list):
        s = ', '.join([fmt_int(x) for x in xx])
        return f'[{s}]' 
    return str(int(xx))
         