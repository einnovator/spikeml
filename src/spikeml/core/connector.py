
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union



from spikeml.utils.vector import _sum, upsample
from spikeml.core.base import Component, Module, Fan, Composite, Chain, Adapter
from spikeml.core.params import Params, LayerParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

from spikeml.core.connector_monitor import ConnectorMonitor, RateConnectorMonitor, LIConnectorMonitor
from spikeml.core.connector_viewer import  ConnectorMonitorViewer, RateConnectorMonitorViewer, LIConnectorMonitorViewer

from spikeml.core.matrix import matrix_split, normalize_matrix, _mult, cmask, cmask2, matrix_init, matrix_init2
from spikeml.utils.nb_util import xdisplay, Markup
from spikeml.core.signal import Source

class Connector(Component):
    """
    Neural Connections base class.
    """

    def __init__(self, params=None,
        auto_sample=True, monitor=None, viewer=None, name=None, callback=None):
        super().__init__(name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer, callback=callback)
        
    def step(self, s, y, context: Optional[Any]=None):
        y = self.propagate(s, y, context)
        return y

    def __call__(self, s, y, context: Optional[Any]=None):
        return self.step(s, y, context)

    def is_batch(self, context: Optional[Any]=None):
        return context is not None and context.batch
    
    def propagate(self, s, y, context: Optional[Any]=None):
        return None

class LinearConnector(Connector):
    """
    Linear Connector. Connection weight are static. Sub-class implement specific update rules and dynamics.
    """

    def __init__(self, M=None, params=None,
        monitor=True, viewer=None, name=None, callback=None):
        super().__init__(params=params, name=name, callback=callback)
        if params is None:
            params = ConnectorParams()
        self.params = params
        self.M = self._init_matrix(M, params)
        self.shape =self.M.shape if self.M is not None else None 
        if monitor==True:
            monitor = LIConnectorMonitor(ref=self)
        if viewer==True:
            viewer = LIConnectorMonitorViewer(monitor)
        self.viewer=viewer
        self.monitor=monitor

    def _init_matrix(self, M, params):
        if isinstance(M, tuple) or isinstance(M, (int, float)) and not isinstance(M, bool):
            M = matrix_init(size=M, params=params)
        return M

    def render(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().render(options)

    #def __repr__(self):
    #    return f"{type(self).__name__}({self.M!r})"
        
    def __matmul__(self, other):
        if hasattr(other, 'M'):
            other = other.M
        return self.M @ other
        
    def __add__(self, other):
        return self.M + other.M

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.M - other.M

    def __rsub__(self, other):
        return other.M - self.M

    def __rmatmul__(self, other):
        if isinstance(other, C):
            return C(other.M @ self.M)
        return C(other @ self.M)

    # Multiplication (*)
    def __mul__(self, other):
        if hasattr(other, 'M'):
            other = other.M
        return self.M * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if hasattr(other, 'M'):            
            return np.array_equal(self.M, other.M)
        return False
    
    @property
    def T(self):
        return self.M.T        

    def propagate(self, s, y, context: Optional[Any]=None):
        self._M = self.M
        return self.M
    
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        _s = f'{self.name}.' if self.name is not None else ''
        xdisplay(Markup(f'{_s}M', self.M))
        
        
class RateConnector(LinearConnector):
    """
    Linear Connector with rate-based update rules.
    """

    def __init__(self, M=None, params=None, 
        monitor=True, viewer=True, name=None, callback=None):
        super().__init__(M=M, params=params, monitor=monitor, viewer=viewer, name=name, callback=callback)
        if monitor==True:
            monitor = RateConnectorMonitor(ref=self)
        if viewer==True:
            viewer = RateConnectorMonitorViewer(monitor)
        self.monitor= monitor
        self.viewer= viewer
        self._M = self.M 
        self.Zp = self.Zn = None
        self.dM = self.dMp = self.dMn = None
        
    def propagate(self, zs, zy, context: Optional[Any]=None):
        self._M = self.M.copy()
        self.dMp = self.dMn = None 
        if not self.is_batch(context):
            if self.params.t_p>0:
                self.Zp = np.outer(zy, zs)
                self.dMp = (1/self.params.t_p)*(self.Zp)
            if self.params.t_d>0: 
                self.Zn = np.outer(zy, self.params.vmax-zs)
                self.dMn = -(1/self.params.t_d)*(self.Zn)
        else:        
            if self.params.t_p>0:
                Zp = zy[:, :, None] * zs[:, None, :]
                dMp = (1/self.params.t_p)*Zp 
                self.dMp = np.sum(dMp, 0)
            if self.params.t_d>0:
                Zn = zy[:, :, None] * (self.params.vmax - zs[:, None, :])
                dMn = (1/self.params.t_d)*Zn 
                self.dMn = -np.sum(dMn, 0)
                
        self.dM = _sum(self.dMp, self.dMn)

        M = self.M.reshape(self.M.shape[0], -1) if self.M.ndim>2 else self.M

        print('self.M:', self.M.shape, self.M.ndim, '; M:', M.shape, '; dM:', self.dM.shape)

        M += self.dM

        if self.params.cmin is not None and self.params.cmax is not None:
            M = np.clip(M, self.params.cmin, self.params.cmax)
        if (self.params.c_in is not None and self.params.c_in>0) or (self.params.c_out is not None and self.params.c_out>0):
            M = normalize_matrix(M, c_in=self.params.c_in, c_out=self.params.c_out, strict=False)

        self.M = M.reshape(self.M.shape) if self.M.ndim>2 else M

        return self.M

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        xdisplay(Markup('_M', self._M), Markup('Zp', self.Zp.reshape(self.M.shape)), Markup('Zn', self.Zn.reshape(self.M.shape)), Markup('dM', self.dM.reshape(self.M.shape)), Markup('dMp', self.dMp), Markup('dMn', self.dMn), Markup('M', self.M))


class LIConnector(LinearConnector):
    """
    Leaky-integrate LTP/LTD connections.
    """

    def __init__(self, M=None, params=None, 
                monitor=True, viewer=True, name=None, callback=None):
        super().__init__(M=M, params=params, name=name, callback=callback)
        if self.M is not None:
            self.Cp, self.Cn = np.zeros(self.M.shape), np.zeros(self.M.shape)
            self._Cp, self._Cn = np.zeros(self.M.shape), np.zeros(self.M.shape)
        else:
            self.Cp, self.Cn = None, None
        self._M = M
        self.dM, self.dMp, self.dMn, self.Zp, self.Zn, self.Wp, self.Wn = None, None, None, None, None, None, None 
        if monitor==True:
            monitor = LIConnectorMonitor(ref=self)
        if viewer==True:
            viewer = LIConnectorMonitorViewer(monitor)
        self.monitor=monitor
        self.viewer=viewer
        
    def __matmul__(self, other):
        """Defines self @ other"""
        return self.M @ other
        
    def propagate(self, zs, zy, context: Optional[Any]=None):
        self._M = self.M
        self.M, self.Cp, self.Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn = \
            self.conn_update(self.M, self.Cp, self.Cn, zy, zs, params=self.params, debug=False)
        self.dM, self.dMp, self.dMn, self.Zp, self.Zn, self.Wp, self.Wn = dM, dMp, dMn, Zp, Zn, Wp, Wn 
        return self.M, self.Cp, self.Cn, dM, dMp, dMn, Zp, Zn, Wp, Wn

    def conn_update(self,
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
        Wp = (Cp >= params.k_p).astype(int)
        Wn = (Cn >= params.k_n).astype(int)
        self._Cp[...]  = Cp
        self._Cn[...] = Cn
        Cp -= Wp*params.k_p
        Cn -= Wn*params.k_n
        Cp = np.clip(Cp, 0, None)
        Cn = np.clip(Cn, 0, None)
        if params.t_cp>0:
            a = 1 - 1/params.t_cp
            Cp *= a
        if params.t_cn>0:
            a = 1 - 1/params.t_cn
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


    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        if options is None or options.get('matrix.details', True):
            xdisplay(Markup('_M', self._M), Markup('Cp', self.Cp), Markup('Cn', self.Cn),  Markup('Zp', self.Zp), Markup('Zn', self.Zn), Markup('Wp', self.Wp), Markup('Wn', self.Wn), Markup('dM', self.dM), Markup('dMp', self.dMp), Markup('dMn', self.dMn), Markup('M', self.M))
        else:
            xdisplay(Markup('M', self.M, Markup('Cp', self.Cp), Markup('Cn', self.Cn)))
                
class LIConnector2(LinearConnector):
    def __init__(self, Mp=None, Mn=None, params=None,
        monitor=True, viewer=True, name=None, callback=None):
        super().__init__(params=params, name=name, callback=callback)
        self.Mp = Mp
        self.Mn = Mn

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
            
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        def _m(M):
            return (M[0]+M[1],M[0],M[1]) if type(M)==tuple else M

        xdisplay(Markup('_M', _m(self._M)), Markup('dM', self.dM), Markup('dMp', self.dMp), Markup('dMn', self.dMn), Markup('zc_p', self.zc[0]), Markup('zc_n', self.zc[1]), Markup('M', self.M))


