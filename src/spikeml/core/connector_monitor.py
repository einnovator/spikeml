from typing import Any, List, Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


from spikeml.utils.fmt_utils import fmt_float, fmt_int
from spikeml.utils.vector import normalize_last

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.core.sensor_monitor import SensingMonitor

from spikeml.core.matrix import matrix_split
from spikeml.core.signal import stats_per_input, mean_per_input, sum_per_input, var_per_input, std_per_input


    
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
        
    def _sample(self, context: Optional[Any]=None) -> "ConnectorMonitor":
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
    
class RateConnectorMonitor(ConnectorMonitor):
    """Monitor for RateConnector"""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the connector being monitored.
        """
        super().__init__(ref=ref)
        
    def _sample(self, context: Optional[Any]=None) -> "LIConnectorMonitor":
        """Sample the state of the connector"""
        super()._sample(context)
        self._sample_prop('Zp')
        self._sample_prop('Zn')
        return self
    
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        sx = self._get_sensor_input()
        if sx is None:
            if options is not None and options.get('verbose', False):
                print('WARN: No sensor input:', self)
            return
        print(f'{prefix}.Zp:')
        ref, size, n = sum_per_input(self.Zp, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.Zp, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n)
        print_spike_counts(ref_, ranges, n_)
        print(f'{prefix}.Zn:')
        ref, size, n = sum_per_input(self.Zn, sx, E=self.E)
        ref_, ranges, n_ = sum_per_input(self.Zn, sx, E=self.E, aggregate=False)
        print_spike_counts(ref, size, n)
        print_spike_counts(ref_, ranges, n_)

    
class LIConnectorMonitor(ConnectorMonitor):
    """Monitor for LIConnector"""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the connector being monitored.
        """
        super().__init__(ref=ref)
        
    def _sample(self, context: Optional[Any]=None) -> "LIConnectorMonitor":
        """Sample the state of the connector"""
        super()._sample(context)
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
            if options is not None and options.get('verbose', False):
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

