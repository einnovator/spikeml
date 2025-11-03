
import numpy as np

def fmt_float(xx, d=4):
    if isinstance(xx, np.ndarray):
        xx = xx.tolist()
    fmt = "{:.{}f}"
    if isinstance(xx, list):
        s = ', '.join([fmt_float(x, d) for x in xx])
        return f'[{s}]' 
    return fmt.format(xx, d) 
         
def fmt_int(xx):
    if isinstance(xx, np.ndarray):
        xx = xx.tolist()
    if isinstance(xx, list):
        s = ', '.join([fmt_int(x) for x in xx])
        return f'[{s}]' 
    return str(int(xx))
         