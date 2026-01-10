
from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import math
from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer
from spikeml.core.signal import signal_changes

from spikeml.core.sensor_viewer import SensingMonitorViewer

from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix
from spikeml.utils.filter_util import filter, filter_count


class SNNMonitorViewer(SensingMonitorViewer):
    """Viewer for SNNMonitor, visualizing spikes, membrane potential, and outputs."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: SNNMonitor object to visualize.
        """
        super().__init__(monitor)


    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render SNN monitor data including spikes, membrane potentials, outputs."""
        tt = self._signal_changes()
        ref = self.get_ref()
        if filter('x', options, ref):
            _,axs = self._axes(2)        
            self._plot_xt(['x'], options=options, ax=axs[0])
            self._plot_lidata('x', self.monitor.ref.params.k_x, options=options, ax=axs[1])
            plt.show()
        K = filter_count(['zy', 'y', 'y', 'r'], options, ref)
        if K>0:
            _,axs = self._axes(K)        
            k = 0
            if filter('zy', options, ref):
                self._plot_spikes('zy', tt=tt, callback=lambda ax: self._plot_input(va='top', ax=ax), options=options, ax=axs[k])
                k += 1
            if filter('y', options, ref):
                self._plot_xt(['y'], options=options, ax=axs[k])
                self._plot_data(['y'], options=options, ax=axs[k+1], callback=lambda ax: self._plot_input(ax))
                k += 2
            if filter('r', options, ref):
                self._plot_data(['r'], options=options, ax=axs[k], callback=lambda ax: self._plot_input(ax))
                k += 1
            plt.show()
        if filter('k_x', options, ref):
            _,axs = self._axes(2)        
            self._plot_xt(['k_x'], options=options, ax=axs[0])
            self._plot_data(['k_x'], options=options, ax=axs[1], callback=lambda ax: self._plot_input(ax))
            plt.show()
        if filter('u', options, ref):
            self._plot_data(['u', 'us'], shared=True, callback=lambda ax: self._plot_input(ax), options=options)

class SSNNMonitorViewer(SensingMonitorViewer):
    """Viewer for SSNNMonitor, visualizing stochastic spiking layers."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: SSNNMonitor object to visualize.
        """
        super().__init__(monitor)

        
    def render(self, options=None):
        tt = self._signal_changes()
        ref = self.get_ref()
        K = filter_count(['zy', 'y', 'y'], options, ref)
        if K>0:
            _,axs = self._axes(K)        
            k = 0
            if filter('zy', options, ref):
                self._plot_spikes('zy', tt=tt, callback=lambda ax: self._plot_input(ax=ax), options=options, ax=axs[k])
                k += 1
            if filter('y', options, ref):
                self._plot_xt(['y'], options=options, ax=axs[k])
                self._plot_data(['y'], options=options, ax=axs[k+1], callback=lambda ax: self._plot_input(ax))
                k += 2
            #if filter('r', options, ref):
            #    self._plot_data(['r'], options=options, ax=axs[k], callback=lambda ax: self._plot_input(ax))
            #    k += 1
            plt.show()
        if filter('u', options, ref):
            self._plot_data(['u', 'us'], shared=True, callback=lambda ax: self._plot_input(ax), options=options)
