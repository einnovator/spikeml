import numpy as np

from spikeml.core.params import SSNNParams
from spikeml.core.feedback import compute_error, compute_sg

from spikeml.core.spikes import pspike, spike, plot_pspike

def test_compute_error():
    params = SSNNParams()
    print(params.fmt())
    params = SSNNParams(g=1, e_err=5, pmax=1, e_z=2)
    print(params.fmt())
    data = {'sx': [], 'y': [], 'err': [], 'sg': [], 's': [], 'sm': [], 'ps': [], 'zs': []}
    def _err(s, y):
        err = compute_error(s, y)
        sg = compute_sg(err, params)
        s =  np.clip(sx + params.g*y, params.vmin, params.vmax).round(2)
        sm = np.clip(s*sg, params.vmin, params.vmax).round(2)
        ps = pspike(sm, params).round(2)
        zs = spike(sm, params)
        data['sx'].append(str(sx))
        data['y'].append(str(y))
        data['err'].append(err)
        data['sg'].append(sg)
        data['s'].append(str(s))
        data['sm'].append(str(sm))
        data['ps'].append(str(ps))
        data['zs'].append(str(zs))
        print(f'sx: {sx} y: {y}', f'-> err: {err:.2f}', f'; sg: {sg:.2f}', f'=> s: {s}', f'=> sm: {sm}', f'=> ps: {ps} ; zs: {zs}')
    for A in [.1, .3, .5]:
        sx = np.array([A,0.0])
        _err(sx, np.array([0.0,0.0]))
        _err(sx, sx)
        _err(sx, np.array([1.0,0]))
        _err(sx, np.array([0,1.0]))
        _err(sx, np.array([.5,0]))
        _err(sx, np.array([0,.5]))
        _err(sx, np.array([.5,.5]))

    df = pd.DataFrame(data)
    return df

def test_compute_error2():
    params = SSNNParams()
    data = {'f': [], 'R': [], 's': [], 'y': [], 'err': []}
    def _add(f, s, y, R, err, debug=True):
        data['f'].append(f)
        data['s'].append(s)
        data['y'].append(y)
        data['R'].append(R)
        data['err'].append(err)
        if debug:
            print(f'{f}: R: {R} ; s: {s} ; y: {y}', f'-> err: {err:.2f}')
        
    def _err(s, y, R):
        s_ = np.repeat(s, R)
        y1 = y.reshape(y.shape[0] // R, R).mean(axis=1)
        y2 = y.reshape(y.shape[0] // R, R).sum(axis=1)
        y2 =  np.clip(y2, params.vmin, params.vmax)
        y3 = y.reshape(y.shape[0] // R, R).max(axis=1)
        
        err = xcompute_error(s, y, R=R, method='dp')
        err1 = xcompute_error(s, y, R=R, method='mean')
        err2 = xcompute_error(s, y, R=R, method='sum+clip')
        err3 = xcompute_error(s, y, R=R, method='max')
        print('-'*4)
        _add('dp', s_, y, R, err)
        _add('mean', s, y1, R, err1)
        _add('sum+clip', s, y2, R, err2)
        _add('max', s, y3, R, err3)

               
    _err(np.array([1,0]), np.array([1,0]), R=1)
    _err(np.array([1,0]), np.array([1,1,0,0]), R=2)
    _err(np.array([1,0]), np.array([1,0,0,0]), R=2)
    _err(np.array([1,0]), np.array([0,1,0,0]), R=2)
    _err(np.array([1,0]), np.array([0,0,0,0]), R=2)
    _err(np.array([1,0]), np.array([0,0,1,1]), R=2)
    _err(np.array([1,0]), np.array([0,0,1,0]), R=2)
    _err(np.array([1,0]), np.array([0,0,0,1]), R=2)
    _err(np.array([1,0]), np.array([1,1,1,1]), R=2)
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':  
    #df = test_compute_error()
    #display(df)

    df=test_compute_error2()
    display(df)
