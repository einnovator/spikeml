import numpy as np

from spikelogik.core.params import SSensorParams

from spikelogik.core.spikes import pspike, spike, plot_pspike

def test_spike():
    params = SSensorParams()
    print(params)
    s = np.linspace(params.vmin,params.vmax,num=5)
    ss = s[..., np.newaxis]
    data = []
    for t in range(0,100):
        sz = spike(s, params)
        print(t, s, sz)
        data.append(sz)
        
    plot_spikes(data, title='z', name=None, callback=lambda ax: plot_input(ss,ax=ax))

    
if __name__ == '__main__':  
    test_spike()
