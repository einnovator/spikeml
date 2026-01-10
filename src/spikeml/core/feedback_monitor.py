from typing import Any, List, Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


from spikeml.utils.fmt_utils import fmt_float, fmt_int
from spikeml.utils.vector import normalize_last

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.core.sensor_monitor import SensingMonitor

from spikeml.core.signal import stats_per_input, mean_per_input, sum_per_input, var_per_input, std_per_input


    
class ErrorMonitor(Monitor):
    """Monitor for tracking error and mean error during training."""

    def __init__(self, name: Optional[str] = None,  ref: Optional[Any] = None, E: Optional[int]=0) -> None:
        """
        Args:
            name: Optional name of the monitor.
        """
        super().__init__(name=name, ref=ref)
        self.s = None 
        self.error = None 
        self.mean_error = None
        self._sum_error = 0
        self._n = 0
        self._mean_error = 0
        self.E = E

    def _sample(self, context: Optional[Any]=None) -> "ErrorMonitor":
        """Sample error for a given step.
        
        Args:
            s: Input signal.
            error: Current error value.
            gain: Smoothed signal.
        """
        if context is not None:
            self.compute_err(context.error)
            self.sample_err(context.sx, context.error, context.gain)
        return self

    def compute_err(self, error: float) -> "ErrorMonitor":
        """Update running mean error."""
        self._sum_error += error
        self._n += 1
        self._mean_error = self._sum_error/self._n
        return self

    def sample_err(self, s: np.ndarray, error: float, gain: np.ndarray) -> "ErrorMonitor":
        """Record the current error and associated signals."""
        self._sample_value('s', s)
        self._sample_value('gain', gain)
        self._sample_value('error', error)
        self._sample_value('mean_error', self._mean_error)
        return self

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Print error statistics."""
        prefix = self._prefix()
        print(f'{prefix}:')
        print(f'  ', f'mean_error: {self.mean_error[-1]:.4f}')
        ref, size, means = mean_per_input(self.error, self.s, E=self.E)
        ref, ranges, means_ = mean_per_input(self.error, self.s, E=self.E, aggregate=False)
        for i in range(0, ref.shape[0]):
            print(f'  ', f'{i}: {ref[i]} (#{size[i]}): Err: {fmt_float(means[i], 4)}')
        for i in range(0, ref.shape[0]):
            print(f'  ', f'{i}: {ref[i]} ({ranges[i]}): Err: {fmt_float(means[i], 4)}')

