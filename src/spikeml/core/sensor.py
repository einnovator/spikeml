
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union



from spikeml.utils.vector import _sum, upsample
from spikeml.core.base import Component, Module, Fan, Composite, Chain, Adapter
from spikeml.core.params import Params, LayerParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

from spikeml.core.snn_monitor import SSensorMonitor
from spikeml.core.snn_viewer import  SSensorMonitorViewer

from spikeml.core.layer import Layer

from spikeml.core.spikes import pspike, spike
from spikeml.core.matrix import matrix_split, normalize_matrix, _mult, cmask, cmask2, matrix_init, matrix_init2
from spikeml.utils.nb_util import xdisplay, Markup
from spikeml.core.signal import Source


class SSensor(Layer):
    """
    Spike Sensor. 
    """

    def __init__(self, n=None, R=1,
                 phase: Optional[Any] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = False,
                 monitor: Union[bool, Any] = True,
                 viewer: Union[bool, Any] = True,
                 name: Optional[str] = None,
                 callback: Optional[Any] = None) -> None:
        super().__init__(phase=phase, name=name, auto_sample=auto_sample, callback=callback)
        if n is None:
            n = params.size
        self.sx = np.zeros(n)
        self.R = R
        self.s = np.zeros(n*R)
        self.zs = np.zeros(self.s.shape[0])
        self._s = np.zeros(self.s.shape[0])
        self._sx = np.zeros(self.s.shape[0])
        self.b = np.zeros(self.s.shape[0])

        self.shape = self.s.shape
        if params is None:
            params = SSensorParams()
        self.params = params
        if monitor==True:
            monitor = SSensorMonitor(ref=self)
        if viewer==True:
            viewer = SSensorMonitorViewer(monitor)
        self.viewer=viewer
        self.monitor=monitor
                
    def propagate(self, s, context: Optional[Any]=None):
        if self.is_propagation(context):
            s,sx = (s[0],s[1]) if isinstance(s, tuple) else (s,s) 
            R = self.s.shape[0] // s.shape[0]                    
            self.R = R
            self._s = s
            self._sx = sx
            self.s = upsample(s, R)
            self.sx = upsample(sx, R)
            s_ = self.s
            if self.b is not None:
                s_ -=  self.b
            self.zs = spike(s_, self.params)
            
            self.b = bias_update(self.b, s_, params=self.params)        

        return self.s,self.zs

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        _s = f'(_sx={self._sx}; _s={self._s}) ; ' if self.R>1 else ''
        print(f'{self.name}: {_s}sx={self.sx}; s={self.s}  -> zs={self.zs}') 
