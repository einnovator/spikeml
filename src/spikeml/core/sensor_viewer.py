
from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import math
from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer
from spikeml.core.signal import signal_changes


from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix
from spikeml.utils.filter_util import filter, filter_count

class SensingMonitorViewer(MonitorViewer):
    """Base Viewer for input marker visualization."""

    def __init__(self, monitor: Monitor, E=0) -> None:
        """
        Args:
            monitor: Monitor object to visualize.
            E : float, optional
                Tolerance for considering two signal vectors equivalent.
                Default is 0 (exact match).
        """
        super().__init__(monitor)
        self.E = E
  
    def _get_sensor_input(self) -> Optional[np.ndarray]:
        from spikeml.core.sensor import SSensor
        sx = self._get('sx')
        if sx is None:
            sx = self._get('s')
            if sx is not None:
                return sx 
        if self.monitor is not None and not isinstance(self.monitor.ref, SSensor):
            """Retrieve sensor input 'sx' from connected SSensor layer."""
            _parent = getattr(self.monitor.ref, '_parent', None)
            if _parent is not None:
                _parent = getattr(_parent, '_parent', _parent)  
                if hasattr(_parent, 'find'):         
                    sensor = _parent.find(SSensor)
                    if sensor is not None and sensor.monitor is not None:
                        sx = getattr(sensor.monitor, 'sx', getattr(sensor.monitor, 's', None))
                        return sx
        return None

    def _signal_changes(self, E= None) -> np.ndarray:
        """Get time of changes on sensor input."""
        sx = self._get_sensor_input()
        if sx is None:
            return None
        if E is None: 
            E = self.E
            if E is None:
                E = 0
        return signal_changes(sx, E)
    
    def _plot_input(self, ax: plt.Axes, va: Optional[str] = None) -> None:
        """Plot sensor input on the provided axis."""
        sx = self._get_sensor_input()
        if sx is not None:
            plot_input(sx, va=va, ax=ax)


class SSensorMonitorViewer(SensingMonitorViewer):
    """Viewer for SSensorMonitor, visualizing sensor spikes and input."""

    def __init__(self, monitor: Monitor, E=0) -> None:
        """
        Args:
            monitor: SSensorMonitor object to visualize.
            E : float, optional
                Tolerance for considering two signal vectors equivalent.
                Default is 0 (exact match).
        """
        super().__init__(monitor, E=E)

            
    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render the SSensor monitor data including spikes and sensor input."""
        ref = self.get_ref()
        K = filter_count(['sx', 'sx', 's', 's', 'zs'], options, ref)
        if self._get('sx') is None:
            K -= 2
        if K>0:
            _,axs = self._axes(K)
            k = 0
            if filter('sx', options, ref) and self._get('sx') is not None:
                self._plot_xt(['sx'], options=options, ax=axs[0])
                self._plot_data(['sx'], callback=lambda ax: self._plot_input(ax=ax), options=options, ax=axs[1]) 
                k += 2
            if filter('s', options, ref):
                self._plot_xt(['s'], options=options, ax=axs[0+k])
                self._plot_data(['s'], callback=lambda ax: self._plot_input(ax=ax), options=options, ax=axs[1+k]) 
                k += 2
            if filter('zs', options, ref):
                self._plot_spikes('zs', callback=lambda ax: self._plot_input(ax=ax, va='top'), options=options, ax=axs[k])                
                k += 1
            plt.show()