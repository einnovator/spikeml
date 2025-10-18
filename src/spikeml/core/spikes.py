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
      
def plot_pspike():
    params = SSNNParams()
    print(params)
    ss = np.stack([np.linspace(params.vmin,params.vmax,num=20)],axis=1)
    #print(ss)
    for e_z in np.linspace(1,5,num=5):
        data = []
        params = SSNNParams(e_z=e_z)
        #print(params)
        x = []
        for s in ss:
            p = pspike(s, params)
            #print(s, p)
            x.append(s[0])
            data.append(p)
        plt.plot(x, data, label=f'e_z={e_z}')
    plt.legend()
    plt.show()
    
