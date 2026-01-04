
import numpy as np
import numbers
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union



from spikeml.utils.vector import _sum, upsample
from spikeml.core.base import Component, Module, Fan, Composite, Chain, Adapter
from spikeml.core.params import Params, LayerParams

from spikeml.core.snn_monitor import LayerMonitor
from spikeml.core.snn_viewer import  LayerMonitorViewer

from spikeml.utils.nb_util import xdisplay, Markup
from spikeml.core.signal import Source
from spikeml.core.connector import Connector

from spikeml.utils.vector import normalize_all, _sum


PHASE_PROPAGATION=0
PHASE_LEARNING=1

def bias_update(
    b: np.ndarray,
    y: np.ndarray,
    params: Any,
    batch: Optional[bool] = False,
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
    params : SSNNParams
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

    dp = y_ * (1/params.t_b) if params.t_b>0 else None 
    dn = - y0_ * (1/(params.t_bn)) if params.t_bn>0 else None
    db = _sum(dp, dn)
    if db is not None:
        if batch:
            db = db.sum(axis=0)
        b_ = b + db
        b_[b_<0] = 0
    else:
        b_ = b
    if debug:
        print(f'b : {b} ({params.t_b}, {params.t_bn})')
        print(f'y : {y}')
        print(f'y_: {y_}')
        print(f'y0_: {y0_}')
        print(f'dp:{dp}')
        print(f'dn: {dn}')
        print(f'db: {db}')
        print(f'b->{b_}')

    return b_

class Layer(Module):
    """
    Base class for neural layers.

    Attributes:
        name: Optional name of the layer.
        phase: Phase where layer is active. If None implies all. (Default: None)
        params: Parameters associated with the layer.
        auto_sample: Whether to automatically sample during updates.
        monitor: Optional monitor object for logging or visualization.
        viewer: Optional viewer object for visualization.
        callback: Optional callback function.
    """
    def __init__(self,
                 phase: Optional[Any] = None,
                 name: Optional[str] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Optional[Any] = None,
                 viewer: Optional[Any] = None,
                 callback: Optional[Any] = None):
        super().__init__(name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer, callback=callback)
        self.phase = phase

    def is_propagation(self, context: Optional[Any]=None):
        return context is None or context.phase is None or context.phase==PHASE_PROPAGATION

    def is_learning(self, context: Optional[Any]=None):
        return context is None or context.phase is None or context.phase==PHASE_LEARNING



class SimpleLayer(Layer):
    """
    Base class for layers with an internal connector or matrix `M`.

    Attributes:
        M: Internal connector or matrix representing weights or connectivity.
        shape: Shape of the connector or matrix `M`.
        n: Number of neurons (first dimension of `M`).
    """

    def __init__(self,
                 M: Optional[Any] = None,
                 phase: Optional[Any] = None,
                 name: Optional[str] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Optional[Any] = None,
                 viewer: Optional[Any] = None,
                 callback: Optional[Any] = None) -> None:
        super().__init__(phase=phase, name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer, callback=callback)
        self.M = M
        if isinstance(self.M, Connector):
            self.M._parent = self
        self.shape = self.M.shape if M is not None else None
        self.n = None
        if self.shape is not None:
            self.n = self.shape[0]
        
    def collect(self, criteria: Union[type, str], out: list[Component] = None) -> list[Component]:
        """
        Collect components by type, name.

        Parameters
        ----------
        criteria : type | str | Module
            Criteria to match component (type, name).

        Returns
        -------
        list[Component]
            List of matching components.
        """
        if out is None:
            out = []
        super().collect(criteria, out)
        if self.M and getattr(self.M,'collect') is not None:
            self.M.collect(criteria, out)
        return out
   

    def render(self, options: Optional[dict] = None) -> None:
        """
        Render the layer and its matrix if available.
        
        Args:
            options: Optional dictionary of rendering options.
        """
        super().render(options)
        if self.M is not None:
            if options is None or options.get('render.matrix', True):
                if isinstance(self.M, Connector):
                    self.M.render(options)

    def sample(self, context: Optional[Any]=None) -> None:
        """
        Sample the internal state of the layer and its matrix if available.
        """
        super().sample(context)
        if isinstance(self.M, Connector):
            self.M.sample(context)
            
    def post_step(self, s: Any, y: Any) -> None:
        """
        Invoke registered callbacks after a computation step.

        Parameters
        ----------
        s : any
            Input signal or state for the current step.
        y : any
            Output signal or state from the current step.
        """
        super().post_step(s, y)
        if isinstance(self.M, Connector):
            self.M.post_step(s, y)

    def log_connector(self, options: Optional[Dict[str, Any]] = None):
        if options is None or options.get('log.matrix', True):
            if hasattr(self.M, 'log'):
                self.M.log(options)
            elif self.M is not None:
                _s = f'{self.name}.' if self.name is not None else ''
                xdisplay(Markup(f'{_s}M', self.M))

    def log_monitor(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Log monitor.

        Parameters
        ----------
        options : dict, optional
            Additional logging configuration.
        """
        super().log_monitor(options)
        self.log_connector_monitor(options)

    def log_connector_monitor(self, options: Optional[Dict[str, Any]] = None) -> None:
        if hasattr(self.M, 'log_monitor'):
            self.M.log_monitor(options)

class LinearLayer(SimpleLayer):
    """
    Linear layer.
    
    Attributes:
        M: Connector or matrix
        batch: batch mode.
        s, zs: Last Input and spike output.
        y, zy: Last Output potentials and output spikes.
        params: Layer parameters.
        monitor: Optional monitoring object.
        viewer: Optional viewer object.
    """

    def __init__(self,
                 M: Optional[Any] = None,
                 b: Optional[Any] = 0,
                 batch: Optional[bool] = True,
                 phase: Optional[Any] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Union[bool, Any] = True,
                 viewer: Union[bool, Any] = True,
                 name: Optional[str] = None,
                 callback: Optional[Any] = None) -> None:
        super().__init__(M=M, phase=phase, name=name, auto_sample=auto_sample, callback=callback)
        self.s,self.zs = None,None
        self.y,self.zy = None,None
        if isinstance(b, numbers.Number):
            b = np.ones((M.shape[0]))*b
        self.b = b
        self.batch = batch
        self.r = None
        if params is None:
            params = LayerParams()
        self.params = params
        if monitor==True:
            monitor = LayerMonitor(ref=self)
        if viewer==True:
            viewer = LayerMonitorViewer(monitor)
        self.viewer=viewer
        self.monitor=monitor
                
    def propagate(self, s: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], context: Optional[Any]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate input spikes through the network and update neuron potentials.
        
        Args:
            s: Input array or tuple (s, zs) of vector pair (rate, spikes).

        Returns:
            Tuple containing updated potentials (y) and spikes (zy).
        """
        if self.is_propagation(context):
            s, zs = s if isinstance(s, tuple) else (s, None)
            self.s, self.zs = s, zs

            self.y = self._propagate(s)
            
            if zs is not None:
                self.zy = self._propagate(zs)


            if self.b is not None:
                self.b = bias_update(self.b, self.y, batch=self.batch, params=self.params)        

        if self.is_learning(context):
            if hasattr(self.M, 'propagate'):    
                zs, zy = (self.zs,self.zy) if self.zs is not None else (self.s, self.y)   
                self.M.propagate(zs, zy, context)

        
        return self.y,self.zy if self.zy is None else self.y 


    def _propagate(self, x):
        if not self.batch:
            _y = self.M @ x
        else:
            _y = x @ self.M.T
        y = np.clip(_y, self.params.vmin, self.params.vmax)
        if self.b is not None:
            _y -= self.b
        y = np.clip(_y, self.params.vmin, self.params.vmax)
        return y

    def _reshape_input(self, x):
        if not self.batch:
            x = x.flatten()
        else:
            x = x.reshape(x.shape[0], -1)
        return x


    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        if self.zs is None:
            print(f'{self.name}: s={self.s} -> y={self.y}') 
        else:
            print(f'{self.name}: s={self.s} | zs={self.zs} -> y={self.y} | zy={self.zy}') 

        self.log_connector(options)


class NormalizeLayer(Layer):
    """
    NormalizeLayer 
    """

    def __init__(self,
                norm: Optional[int] = 2,
                scale: Optional[float] = 1,
                name: Optional[str] = None,
                 callback: Optional[Any] = None) -> None:
        super().__init__(name=name)
        self.norm = norm
        self.scale = scale
                
    def propagate(self, s, context: Optional[Any]=None):
        s_ = normalize_all(s, norm=self.norm)
        if self.scale is not None and self.scale!=1:
            s_ = s_ * self.scale
        return s_

    
class ThresholdLayer(Layer):
    """
    Threshold Sensor. 
    """

    def __init__(self,
                b: Optional[Any] = 0,
                 name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.b = b
                
    def propagate(self, s, context: Optional[Any]=None):
        if not isinstance(s, tuple):
            s = self._propagate(s)
            return s
        else:
            s,sx = (s[0],s[1])
            s = self._propagate(s)
            sx = self._propagate(sx)
            return s,sx 
    
    def _propagate(self, s):
        if self.b is not None:
            s[s < self.b] = self.b
            #zs = (s > self.b).astype(np.int8)
        else:
            zs = s
        return zs


class BinaryThresholdLayer(Layer):
    """
    BinaryThreshold Sensor. 
    """

    def __init__(self,
                b: Optional[Any] = 0,
                 name: Optional[str] = None) -> None:
        super().__init__(phase=phase, name=name)
        self.b = b
                
    def propagate(self, s, context: Optional[Any]=None):
        if not isinstance(s, tuple):
            s = self._propagate(s)
            return s
        else:
            s,sx = (s[0],s[1])
            s = self._propagate(s)
            sx = self._propagate(sx)
            return s,sx 

    def _propagate(self, s):
        if self.b is not None:
            s = (s > self.b).astype(np.int8).astype(float)
        return s


