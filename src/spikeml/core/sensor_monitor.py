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

    def _sample(self, context: Optional[Any]=None) -> None:
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
        from spikeml.core.sensor import SSensor
        
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
        else:
            if options is not None and options.get('verbose', False):
                print('WARN: no parent:', self.ref)            
        return None
    
    def _log(self, options: Optional[Dict[str, Any]] = None) -> None:
        sx = self._get_sensor_input()
        if sx is None:
            if options is not None and options.get('verbose', False):
                print('WARN: No sensor input', self)
            return
        ref, size, n = sum_per_input(self.zy, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.zy, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n, prob=True, soft_prob=True)
        print_spike_counts(ref_, ranges, n_, prob=True, soft_prob=True)

