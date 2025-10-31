import numpy as np

from spikeml.core.params import SSensorParams

from spikeml.core.spikes import pspike, spike

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
    
    
if __name__ == '__main__':  
    test_spike()
    plot_pspike()
_