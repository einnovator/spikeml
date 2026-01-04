
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union



from spikeml.utils.vector import _sum, upsample
from spikeml.core.base import Component, Module, Fan, Composite, Chain, Adapter
from spikeml.core.params import Params, BiasParams, LayerParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

from spikeml.core.snn_monitor import LayerMonitor, SNNMonitor, SSNNMonitor, ConnectorMonitor, RateConnectorMonitor, LIConnectorMonitor
from spikeml.core.snn_viewer import  LayerMonitorViewer, SNNMonitorViewer, SSNNMonitorViewer, ConnectorMonitorViewer, RateConnectorMonitorViewer, LIConnectorMonitorViewer

from spikeml.core.spikes import pspike, spike
from spikeml.core.matrix import matrix_split, normalize_matrix, _mult, cmask, cmask2, matrix_init, matrix_init2
from spikeml.utils.nb_util import xdisplay, Markup
from spikeml.core.signal import Source

from spikeml.core.layer import SimpleLayer, LinearLayer, bias_update
from spikeml.core.connector import Connector, LinearConnector, RateConnector, LIConnector

def __cov_update(M, s=None, y=None, zc_p=None, zc_n=None, params=None, debug=False):
    s = s
    y = y
    #LTP
    dMp = None
    if params.t_p>0:
        if zc_p is not None:
            dMp = zc_p
        else:
            dMp = np.outer(y, s) 
            #dMp = dMp*zc_p
        dM = dMp
        dM = (1/params.t_p)*(dM)
    else:
        dM = None
    dMn = None
    if params.t_d>0: #LTD
        #print(f'sn: {sn}')
        if zc_n is not None:
            dMn = zc_n
        else:
            k_ltd_ = params.vmin+(params.vmax-params.vmin)*params.k_ltd
            sn = params.vmax-s
            sn = np.clip(sn, params.vmin, params.vmax)
            sn[sn < (1-k_ltd_)] = 0
            dMn = -np.outer(y, sn)
            #dMn = dMn*zc_n
            dMn *= params.f_ltd
        dMn = (1/params.t_d)*(dMn)
        dM = dMp+dMn if dM is not None else dMn

    #print(f't_p: {t_p}; dM: {dM}')
    #d,_,_=cmask2(M, c_in, c_out)
    #if d is not None:
    #    dM *= d
    _M = M
    if dM is not None:
        M = M + dM
    M_ = M
    if params.cmin is not None and params.cmax is not None:
        M_ = np.clip(M_, params.cmin, params.cmax)
    if (params.c_in is not None and params.c_in>0) or (params.c_out is not None and params.c_out>0):
        M_ = normalize_matrix(M_, c_in=params.c_in, c_out=params.c_out, strict=False)

    if debug:
        xdisplay(Markup('_M', _M), Markup('dM', dM), Markup('dMp', dMp), Markup('dMn', dMn), Markup('M', M), Markup('M_', M_))

    return M_, dM, dMp, dMn



def conn_update(
    M: np.ndarray,
    Cp: Optional[np.ndarray],
    Cn: Optional[np.ndarray],
    zy: np.ndarray,
    zs: np.ndarray,
    params: Optional['SSNNParams'] = None,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update a connection matrix using LTP/LTD rules.

    Parameters
    ----------
    M : np.ndarray
        Current connection matrix.
    Cp : np.ndarray or None
        Positive connection accumulator matrix. If None, initialized to zeros.
    Cn : np.ndarray or None
        Negative connection accumulator matrix. If None, initialized to zeros.
    zy : np.ndarray
        Post-synaptic spike vector.
    zs : np.ndarray
        Pre-synaptic spike vector.
    params : SSNNParams, optional
        Parameters containing thresholds, decay times, LTP/LTD settings, etc.
    debug : bool
        Whether to print debugging information.

    Returns
    -------
    Tuple containing:
        M_ : np.ndarray
            Updated and normalized connection matrix.
        Cp : np.ndarray
            Updated positive accumulator.
        Cn : np.ndarray
            Updated negative accumulator.
        dM : np.ndarray or None
            Net change in weights.
        dMp : np.ndarray or None
            Positive weight updates (LTP).
        dMn : np.ndarray or None
            Negative weight updates (LTD).
        Zp : np.ndarray
            Positive Hebbian contribution.
        Zn : np.ndarray
            Negative Hebbian contribution.
        Wp : np.ndarray
            Positive weight mask.
        Wn : np.ndarray
            Negative weight mask.
    """
    if params is None:
        params = SSNNParams()
    if Cp is None:
        Cp = np.zeros(M.shape)
    if Cn is None:
        Cn = np.zeros(M.shape)
    Zp = np.outer(zy, zs)
    Zn = np.outer(zy, 1-zs)
    Cp += Zp
    Cn += Zn
    Wp = (Cp >= params.c_k).astype(int)
    Wn = (Cn >= params.c_k).astype(int)
    Cp -= Wp*params.c_k
    Cn -= Wn*params.c_k
    Cp = np.clip(Cp, 0, None)
    Cn = np.clip(Cn, 0, None)
    if params.t_c>0:
        a = 1 - 1/params.t_c
        Cp *= a
        Cn *= a
    dMp = (1/params.t_p)*(Wp) if params.t_p>0 else None #LTP
    dMn = -(1/params.t_d)*(Wn) if params.t_d>0 else None #LTD
    dM = _sum(dMp, dMn)
    _M = M
    if dM is not None:
        M = M + dM
    M_ = M
    if params.cmin is not None and params.cmax is not None:
        M_ = np.clip(M_, params.cmin, params.cmax)
    if (params.c_in is not None and params.c_in>0) or (params.c_out is not None and params.c_out>0):
        M_ = normalize_matrix(M_, c_in=params.c_in, c_out=params.c_out, strict=False)

    if debug:
        xdisplay(Markup('_M', _M), Markup('Cp', Cp), Markup('Cn', Cn),  Markup('Zp', Zp), Markup('Zn', Zn), Markup('Wp', Wp), Markup('Wn', Wn), Markup('dM', dM), Markup('dMp', dMp), Markup('dMn', dMn), Markup('M', M), Markup('M_', M_))

    return M_, Cp, Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn

def ssnn_apply_update(
    M: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    Cp: Optional[np.ndarray],
    Cn: Optional[np.ndarray],
    s: np.ndarray,
    zs: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    params: Optional['SSNNParams'] = None,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply stochastic updates to a (possibly double) spike layer using LTP/LTD rules.

    Parameters
    ----------
    M : np.ndarray or tuple of np.ndarray
        Connection matrix or tuple (Mp, Mn) for positive/negative components.
    Cp : np.ndarray
        Positive accumulator matrix.
    Cn : np.ndarray
        Negative accumulator matrix.
    s : np.ndarray
        Input spikes.
    zs : np.ndarray, optional
        Precomputed input spikes. If None, computed internally.
    b : np.ndarray, optional
        Bias vector.
    params : SSNNParams, optional
        Parameters object containing network hyperparameters.
    debug : bool
        If True, prints debug information.

    Returns
    -------
    Tuple containing:
        y : np.ndarray
            Output values.
        zy : np.ndarray
            Output spikes.
        zs : np.ndarray
            Input spikes.
        M : np.ndarray or tuple
            Updated connection matrix.
        Cp : np.ndarray
            Updated positive accumulator.
        Cn : np.ndarray
            Updated negative accumulator.
        dM : np.ndarray or tuple
            Net weight change.
        dMp : np.ndarray or tuple
            Positive weight change.
        dMn : np.ndarray or tuple
            Negative weight change.
        Zp : np.ndarray
            Positive Hebbian term.
        Zn : np.ndarray
            Negative Hebbian term.
        Wp : np.ndarray
            Positive weight mask.
        Wn : np.ndarray
            Negative weight mask.
    """
    if params is None:
        params = SSNNParams()
        
    _M = M[0]+M[1] if type(M)==tuple else M
    #if w_sd>0:
    #    R = np.random.normal(loc=0, scale=w_sd, size=M.shape)
    #    _M = M + R
        
    _y = _M @ s
    if b is not None:
        _y -=  b
    y = np.clip(_y, params.vmin, params.vmax)
    if zs is None:
        zs = spike(s, params)
    zy = spike(y, params)    
    
    if debug:
        print('s:', s, '; zs:', zs,  '-> y:', y, ' ; zy:', zy)  
        
    if not type(M)==tuple:
        M, Cp, Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn = conn_update(M, Cp, Cn, zy, zs, params=params, debug=debug)
    else:
        Mp,Mn = M
        M, Cp, Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn = conn_update(Mp, Cp, Cn, zy, zs, params=params, debug=debug)
        Mp[Mp < 0] = 0
        Mn, dMn, dMnp, dMnn, Zp, Zn, Wp, Wn = cov_update(Mn, Cp, Cn, zy, zs, params=params, debug=debug)
        Mn[Mn > 0] = 0
        M_=(Mp,Mn)
        dM=(dMp,dMn)
        dMp=(dMpp,dMpn)
        dMn=(dMnp,dMnn)

    return y, zy, zs, M, Cp, Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn


def __bias_update(
    b: np.ndarray,
    y: np.ndarray,
    params: 'BiasParams',
    debug: bool = False
) -> np.ndarray:
    """
    Update the adaptive bias for a stochastic spike layer.

    Parameters
    ----------
    b : np.ndarray
        Current bias vector.
    y : np.ndarray
        Output of the layer.
    params : BiasParams
        Parameters containing adaptive threshold settings.
    debug : bool
        If True, prints debug info.

    Returns
    -------
    np.ndarray
        Updated bias vector.
    """
    if params.t_b<=0:
        return b
    y_ = ((y-params.vmin)/(params.vmax-params.vmin))**params.e_b
    y0_ = ((params.vmax-y)/(params.vmax-params.vmin))**params.e_b

    b_ = b + (y_-y0_) * (1/params.t_b)
    b_[b_<0] = 0
    if debug:
        print(f'b: {b} -> {b_} ({params.t_b}) ; y_={y_}  y0_={y0_}')
    return b_

            
class SNN(SimpleLayer):
    """
    Leaky-Integrator Spiking Neural Network layer with adaptive threshold and random noise.

    Attributes:
        x: Membrane potentials of neurons.
        _x: Internal copy of `x`.
        k_x: Adaptive threshold for each neuron.
        s, zs: Input and spike output.
        y, zy: Output potentials and output spikes.
        r: Random noise added to potentials.
        params: Layer parameters.
        monitor: Optional monitoring object.
        viewer: Optional viewer object.
    """

    def __init__(self,
                 M: Optional[Any] = None,
                 x: Optional[np.ndarray] = None,
                 phase: Optional[Any] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Union[bool, Any] = True,
                 viewer: Union[bool, Any] = True,
                 name: Optional[str] = None,
                 callback: Optional[Any] = None) -> None:
        super().__init__(M=M, phase=phase, name=name, auto_sample=auto_sample, callback=callback)
        self.x = x if x is not None else np.zeros(self.n) if self.n is not None else None
        self._x = x
        self.k_x = np.ones(self.x.shape)*params.k_x if self.x is not None else None
        self.s,self.zs = None,None
        self.y,self.zy = None,None
        self.r = np.zeros(self.x.shape)
        if params is None:
            params = SNNParams()
        self.params = params
        if monitor==True:
            monitor = SNNMonitor(ref=self)
        if viewer==True:
            viewer = SNNMonitorViewer(monitor)
        self.viewer=viewer
        self.monitor=monitor
                
    def propagate(self, s: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], context: Optional[Any]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate input spikes through the network and update neuron potentials.
        
        Args:
            s: Input array or tuple (s, zs) of spikes.

        Returns:
            Tuple containing updated potentials (y) and spikes (zy).
        """
        if self.is_propagation(context):
            s, zs = s if isinstance(s, tuple) else (s, None)
            if zs is None:
                zs = spike(s, self.params)
            self.s, self.zs = s, zs

            y = self.M @ zs
            self.x += y
            r = np.random.normal(loc=0, scale=self.params.r_sd, size=self.x.shape[0])
            self.x += r
            y[y>self.k_x] = 0
            zy = (self.x>self.k_x).astype(int)
            self._x = self.x
            self.x -= zy*self.k_x         
            self.x = np.clip(self.x, 0, None)
            self.x +=  -self.x*1/self.params.t_x    
            self.y, self.zy = y, zy
            self.r = r

            #TODO: update k_x

        if self.is_learning(context):
            if hasattr(self.M, 'propagate'):       
                self.M.propagate(zs, zy, context)

                            
        #if debug:
        #print('s:', s, '; zs:', zs, '->', 'x:', self.x, '; k_x:', self.k_x, ' | y:', y, ' ; zy:', zy)  
        
        return self.y,self.zy

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        print(f'{self.name}: s={self.s} | zs={self.zs} ; r={self.r} -> x={self.x}; k_x={self.x} -> y={self.y} | zy={self.zy}') 

        self.log_connector(options)



class SSNN(SimpleLayer):
    """
    Leaky-Integrator Spiking Neural Network layer with adaptive threshold and random noise.

    Attributes:
        x: Membrane potentials of neurons.
        _x: Internal copy of `x`.
        k_x: Adaptive threshold for each neuron.
        s, zs: Input and spike output.
        y, zy: Output potentials and output spikes.
        r: Random noise added to potentials.
        params: Layer parameters.
        monitor: Optional monitoring object.
        viewer: Optional viewer object.
    """

    def __init__(self,
                 M: Optional[Any] = None,
                 x: Optional[np.ndarray] = None,
                 phase: Optional[Any] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Union[bool, Any] = True,
                 viewer: Union[bool, Any] = True,
                 name: Optional[str] = None,
                 callback: Optional[Any] = None) -> None:
        
        super().__init__(M=M, phase=phase, name=name, auto_sample=auto_sample, callback=callback)
        self.b = np.zeros(self.n)
        self.s,self.zs = None,None
        self.y,self.zy = None,None
        if params is None:
            params = SSNNParams()
        self.params = params
        if monitor==True:
            monitor = SSNNMonitor(ref=self)
        if viewer==True:
            viewer = SSNNMonitorViewer(monitor)
        self.viewer=viewer
        self.monitor=monitor
                
    def __str(self):
        ss = []
        if self.name is not None:
            ss.append(f'{self.name} : ') 
        if self.M is not None:
            ss.append(f'M= {self.M}')
        s = '\n'.join(ss)
        return s


    def propagate(self, s: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], context: Optional[Any]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate input spikes through the network and update neuron potentials.
        
        Args:
            s: Input array or tuple (s, zs) of spikes.

        Returns:
            Tuple containing updated potentials (y) and spikes (zy).
        """
        if self.is_propagation(context):        
            s, zs = s if isinstance(s, tuple) else (s, None)
            if zs is None:
                zs = spike(s, self.params)
            self.s, self.zs = s, zs

            _y = self.M @ s
            if self.b is not None:
                _y -=  self.b
            y = np.clip(_y, self.params.vmin, self.params.vmax)
            zy = spike(y, self.params)
            self.y, self.zy = y, zy

            self.b = bias_update(self.b, self.y, params=self.params)        
    
        if self.is_learning(context):                
            if hasattr(self.M, 'propagate'):
                self.M.propagate(zs, zy, context)
            
        #if debug:
        #   print('s:', s, '; zs:', zs,  '-> y:', y, ' ; zy:', zy, ' ; b:', self.b)  
        
        return self.y,self.zy

    def log(self, options: Optional[dict] = None) -> None:
        """
        Log the current state of the layer.

        Args:
            options: Optional dictionary of logging options.
        """
        print(f'{self.name}: s={self.s} | zs={self.zs} -> y={self.y} | zy={self.zy}') 

        self.log_connector(options)

            



class DSSNN(SSNN):
    """
    Dynamic Stochastic Spike Layer with Adaptive-Threshold
    """

    def __init__(self, M=None, Mx=None, b=None, x=None,
                 phase: Optional[Any] = None,
                params=None, auto_sample=False, monitor=None, viewer=None, name=None, callback=None):
        super().__init__(M=M, b=b, phase=phase, name=name, auto_sample=auto_sample, monitor=None, viewer=None, callback=callback)
        self.x = x if x is not None else np.zeros(self.n, dtype=float)
        if monitor==True:
            monitor = DSSNNMonitor(ref=self)
        if viewer==True:
            viewer = DSSNNMonitorViewer(monitor)
        self.monitor=monitor
        self.viewer=viewer

    def propagate(self, s, context: Optional[Any]=None):
        if self.is_propagation(context):        
            s, zs = s if isinstance(s, tuple) else (s, None)
            if zs is None:
                zs = spike(s, self.params)
            self.s =  s
            
            _y = self.M @ s + self.Mx @ self.x
            if self.b is not None:
                _y -=  self.b
            y = np.clip(_y, params.vmin, params.vmax)
            zy = spike(y, params)    
            
            self.b = bias_update(self.b, self.y, params=params)        

        if self.is_learning(context):        
            self.M.propagate(zs, zy, context)
            self.Mx.propagate(zy, zy, context)
                
        return self.y,self.zy


