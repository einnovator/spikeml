from typing import Any, List, Optional, Union, Dict
import numpy as np


from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.core.sensor_monitor import SensingMonitor


    
class LayerMonitor(SensingMonitor):
    """Generic Monitor for a NN layer."""

    def __init__(self, ref: Optional[Any] = None) -> None:
        """
        Args:
            ref: Reference to the SNN layer being monitored.
        """
        super().__init__(ref=ref)

    def _sample(self, context: Optional[Any]=None) -> None:
        """Sample layer properties and compute derived values."""
        self._sample_prop('y')
        self.compute(context)
        self._sample_prop('u')
        self._sample_prop('us')

    def compute(self, context: Optional[Any]=None) -> None:
        """Compute aggregated values from the layer (e.g., total spikes, outputs)."""
        ref = self.ref
        if context is not None and context.batch:
            ref.u = ref.y.sum(axis=tuple(range(1, ref.y.ndim)))
            ref.ux = ref.x.sum(axis=tuple(range(1, ref.x.ndim)))        
        else:
            ref.u = ref.y.sum()
            ref.ux = ref.x.sum()        

        
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        prefix = self._prefix()
        #print(f'{prefix}.zy:')
        super()._log(options)

