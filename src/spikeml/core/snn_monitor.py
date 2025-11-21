from typing import Any, List, Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


from spikeml.utils.fmt_utils import fmt_float, fmt_int
from spikeml.utils.vector import normalize_last

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.core.matrix import matrix_split
from spikeml.core.signal import stats_per_input, mean_per_input, sum_per_input, var_per_input, std_per_input

def print_spike_counts(ref, ranges, n, prob=False, soft_prob=False):
    for i in range(0, ref.shape[0]):
        ni = np.array(n[i])
        _s = str(ranges[i]) if len(ni.shape)>1 else f'#{ranges[i]}'
        s_ = []
        if prob:
            p = normalize_last(ni)
            s_.append(fmt_float(p, 2))
        if soft_prob:
            p_ = softmax(ni, axis=-1)
            s_.append(fmt_float(p_, 2))
        s_ = ';'.join(s_)
        if len(s_)>0:
            s_ = f' ({s_})'        
        print(f'  {i}: {ref[i]} ({_s}); N:{fmt_int(n[i])}{s_}')

class SSensorMonitor(Monitor):
    """Monitor for SSensor spike sensor layer."""

    def __init__(self, ref: Optional[Any] = None, E: Optional[int]=0) -> None:
        """
        Args:
            ref: Reference to the layer being monitored.
        """
        super().__init__(ref=ref)
        self.E = E

    def sample(self) -> None:
        """Sample properties of the sensor and compute derived values."""

        self._sample_prop('sx')
        self._sample_prop('s')
        self._sample_prop('zs')
        
        self.compute()
        self._sample_prop('us')

    def compute(self) -> None:
        """Compute the sum of spikes in the sensor layer."""
        ref = self.ref
        ref.us = ref.s.sum()
        

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        print(f'{prefix}.zs:')
        ref, size, n = sum_per_input(self.zs, self.sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.zs, self.sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n, prob=True, soft_prob=True)
        print_spike_counts(ref_, ranges, n_, prob=True, soft_prob=True)


class SensingMonitor(Monitor):
    """Base Monitor for sensor-input aware Monitors.

    Attributes
    ----------
    name : Optional[str]
        Name of this monitor instance.
    ref : Optional[Any]
        Reference object whose properties are being monitored.
    """

    def __init__(self, name: Optional[str] = None, ref: Optional[Any] = None, E: Optional[int]=0) -> None:
        super().__init__(name, ref)
        self.E = E

    def _get_sensor_input(self) -> Optional[np.ndarray]:
        from spikeml.core.snn import SSensor
        
        """Retrieve sensor input 'sx' from connected SSensor layer."""
        _parent = getattr(self.ref, '_parent', None)
        if _parent is not None:
            _parent = getattr(_parent, '_parent', _parent)  
            if hasattr(_parent, 'find'):         
                sensor = _parent.find(SSensor)
                #print('!', self, '_parent:', _parent, 'sensor:', sensor)
                if sensor is not None and sensor.monitor is not None:
                    sx = getattr(sensor.monitor, 'sx', None)
                    #print('  !sx:', len(sx) if sx is not None else None)
                    return sx
        return None
    
    def _log(self, options: Optional[Dict[str, Any]] = None) -> None:
        sx = self._get_sensor_input()
        if sx is None:
            print('WARN: No sensor input', self)
            return
        ref, size, n = sum_per_input(self.zy, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.zy, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n, prob=True, soft_prob=True)
        print_spike_counts(ref_, ranges, n_, prob=True, soft_prob=True)


    
class LayerMonitor(SensingMonitor):
    """Generic Monitor for a NN layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SNN layer being monitored.
        """
        super().__init__(ref=ref)

    def sample(self) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('y')
        self._sample_prop('zy')
        self.compute()
        self._sample_prop('u')
        self._sample_prop('us')

    def compute(self) -> None:
        """Compute aggregated values from the layer (e.g., total spikes, outputs)."""
        ref = self.ref
        ref.u = ref.y.sum()
        ref.us = ref.s.sum()        

        
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        print(f'{prefix}.zy:')
        super()._log(options)

class SNNMonitor(SensingMonitor):
    """Monitor for SNN (Spiking Neural Network) layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SNN layer being monitored.
        """
        super().__init__(ref=ref)

    def sample(self) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('x')
        self._sample_prop('y')
        self._sample_prop('zy')
        self._sample_prop('k_x')
        self._sample_prop('r')
        self.compute()
        self._sample_prop('u')
        self._sample_prop('us')

    def compute(self) -> None:
        """Compute aggregated values from the layer (e.g., total spikes, outputs)."""
        ref = self.ref
        ref.u = ref.y.sum()
        ref.us = ref.s.sum()        
        #ref.zym = np.max(ref.zy, axis=1)

        
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        print(f'{prefix}.zy:')
        super()._log(options)


class SSNNMonitor(SensingMonitor):
    """Monitor for SSNN (Stochastic Spiking Neural Network) layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SSNN layer being monitored.
        """
        super().__init__(ref=ref)

    def sample(self) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('y')
        self._sample_prop('zy')
        self._sample_prop('b')

        self.compute()
        self._sample_prop('u')
        self._sample_prop('us')
        
    def compute(self) -> None:
        """Compute aggregated values for the layer (e.g., outputs and spikes)."""
        ref = self.ref
        ref.u = ref.y.sum()
        ref.us = ref.s.sum()
        
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        print(f'{prefix}.zy:')
        super()._log(options)

class ConnectorMonitor(SensingMonitor):
    """Monitor for neural network connectors (synapses)."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the connector being monitored.
        """
        super().__init__(ref=ref)
        self.M = []
        self.dw = []
        self.dwp = []
        self.dwn = []
        self._M = None
        self._Mp = None
        self._Mn = None
        
    def sample(self) -> "ConnectorMonitor":
        """Sample the state of the connector, including weight changes."""
        M = self._get('M')
        if M is not None:
            if type(M)==tuple:
                Mp,Mn = M
                M = Mp+Mn
            else:
                Mp, Mn = matrix_split(M)
            self.M.append(M)
            if self._M is None:
                dw = 0
                dwp = 0
                dwn = 0
            else:
                dM = M-self._M

                dMp = Mp-self._Mp
                dMn = Mn-self._Mn

                dw = float(np.abs(dM).sum())
                dwp = float(dMp.sum())
                dwn = float(Mn.sum())

            self._M = M
            self._Mp = Mp
            self._Mn = Mn
            
            self.dw.append(dw)
            self.dwp.append(dwp)
            self.dwn.append(dw)    
        return self
    

    
class LIConnectorMonitor(ConnectorMonitor):
    """Monitor for LIConnector"""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the connector being monitored.
        """
        super().__init__(ref=ref)
        
    def sample(self) -> "LIConnectorMonitor":
        """Sample the state of the connector"""
        super().sample()
        self._sample_prop('_Cp')
        self._sample_prop('_Cn')
        #self._sample_prop('dM')
        #self._sample_prop('dMp')
        #self._sample_prop('dMn')
        self._sample_prop('Zp')
        self._sample_prop('Zn')
        self._sample_prop('Wp')
        self._sample_prop('Wn')        
        return self
    
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        sx = self._get_sensor_input()
        if sx is None:
            print('WARN: No sensor input:', self)
            return
        print(f'{prefix}.Wp:')
        ref, size, n = sum_per_input(self.Wp, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.Wp, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n)
        print_spike_counts(ref_, ranges, n_)
        print(f'{prefix}.Wn:')
        ref, size, n = sum_per_input(self.Wn, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.Wn, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n)
        print_spike_counts(ref_, ranges, n_)


class ErrorMonitor(Monitor):
    """Monitor for tracking error and mean error during training."""

    def __init__(self, name: Optional[str] = None,  ref: Optional[Any] = None, E: Optional[int]=0) -> None:
        """
        Args:
            name: Optional name of the monitor.
        """
        super().__init__(name=name, ref=ref)
        self.s = None 
        self.err = None 
        self.merr = None
        self._serr = 0
        self._n = 0
        self._merr = 0
        self.E = E

    def sample(self, s: np.ndarray, err: float, sm: np.ndarray) -> "ErrorMonitor":
        """Sample error for a given step.
        
        Args:
            s: Input signal.
            err: Current error value.
            sm: Smoothed signal.
        """
        self.compute_err(err)
        self.sample_err(s, err, sm)
        return self

    def compute_err(self, err: float) -> "ErrorMonitor":
        """Update running mean error."""
        self._serr += err
        self._n += 1
        self._merr = self._serr/self._n
        return self

    def sample_err(self, s: np.ndarray, err: float, sm: np.ndarray) -> "ErrorMonitor":
        """Record the current error and associated signals."""
        self._sample('s', s)
        self._sample('sm', sm)
        self._sample('err', err)
        self._sample('merr', self._merr)
        return self

    def log(self) -> None:
        """Print error statistics."""
        prefix = self._prefix()
        print(f'{prefix}:')
        print(f'  ', f'merr: {self.merr[-1]:.4f}')
        ref, size, means = mean_per_input(self.err, self.s, E=self.E)
        ref, ranges, means_ = mean_per_input(self.err, self.s, E=self.E, aggregate=False)
        for i in range(0, ref.shape[0]):
            print(f'  ', f'{i}: {ref[i]} (#{size[i]}): Err: {fmt_float(means[i], 4)}')
        for i in range(0, ref.shape[0]):
            print(f'  ', f'{i}: {ref[i]} ({ranges[i]}): Err: {fmt_float(means[i], 4)}')

