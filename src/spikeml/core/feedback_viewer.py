    

from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import math
from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer
from spikeml.core.signal import signal_changes


from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix
from spikeml.utils.filter_util import filter, filter_count


class ErrorMonitorViewer(MonitorViewer):
    """Viewer for ErrorMonitor, visualizing error metrics over time."""

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render error metrics and smoothed signals."""
        monitor = self.monitor
        ref = self.get_ref()
        K = filter_count(['error', 'gain'], options, ref)
        if K>0:
            _,axs = self._axes(K)    
            k = 0 
            if filter('error', options, ref):                   
                self._plot_data(['error', 'mean_error'], shared=True, ylim=(0,1.1), options=options, ax=axs[k])
                k += 1
            if filter('gain', options, ref):
                self._plot_data(['gain'], options=options, ax=axs[k])
                k += 1
            plt.show()
