from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.core.matrix import matrix_split

class SSensorMonitor(Monitor):
    """Monitor for SSensor spike sensor layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the layer being monitored.
        """
        super().__init__(ref=ref)

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

class SNNMonitor(Monitor):
    """Monitor for SNN (Spiking Neural Network) layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SNN layer being monitored.
        """
        super().__init__(ref=ref)

    def sample(self) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('sx')
        self._sample_prop('s')
        self._sample_prop('zs')
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

class SSNNMonitor(Monitor):
    """Monitor for SSNN (Stochastic Spiking Neural Network) layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SSNN layer being monitored.
        """
        super().__init__(ref=ref)

    def sample(self) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('sx')
        self._sample_prop('s')
        self._sample_prop('zs')
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

class ConnectorMonitor(Monitor):
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
    
class ErrorMonitor(Monitor):
    """Monitor for tracking error and mean error during training."""

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Args:
            name: Optional name of the monitor.
        """
        super().__init__(name=name)
        self.s = None 
        self.err = None 
        self.merr = None
        self._serr = 0
        self._n = 0
        self._merr = 0

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
        print(f'{prefix}:', f'merr: {self.merr[-1]:.4f}')
        ref, size, means = mean_per_input(self.err, self.s)
        for i in range(0, ref.shape[0]):
            print(f'{prefix}:', f'{i}: {ref[i]} (#{size[i]}); Err: {means[i]:.4f}')

