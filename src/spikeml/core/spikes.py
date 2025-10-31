import numpy as np

def pspike(a, params):
    s = (a-params.vmin)/(params.vmax-params.vmin)
    if isinstance(a, np.ndarray):
        p = params.pf * (1 - np.exp(-s*params.e_z))
    else:
        p = params.pf * (1 - math.exp(-s*params.e_z))
    p = np.clip(p, 0, params.pmax)
    #p = s**params.e_z
    return p


def spike(s, params):
    ps = pspike(s, params)
    ps_ = np.random.random(s.shape[-1]) 
    zs = (ps > ps_).astype(int)
    return zs

