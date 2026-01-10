

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


class ConnectorMonitorViewer(SensingMonitorViewer):
    """Viewer for ConnectorMonitor, visualizing weight matrices and updates."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: ConnectorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render SSNN monitor data including spikes, outputs, and aggregated signals."""
        ref = self.get_ref()
        K = filter_count(['M', 'dw'], options, ref)
        if K>0:
            _,axs = self._axes(K)    
            if filter('M', options, ref):    
                self._plot_mt(['M'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            if filter('dw', options, ref):
                self._plot_data(['dw'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[1])
            plt.show()
        if filter('M', options, ref):                
            self._imshow_nmatrix(['M'], options=options)
        
class RateConnectorMonitorViewer(ConnectorMonitorViewer):
    """Viewer for RateConnectorMonitor, visualizing leaky integrate-and-fire connection dynamics."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: RateConnectorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render LI connector monitor using base connector viewer logic."""
        super().render(options=options)
        ref = self.get_ref()
        return self

class SRateConnectorMonitorViewer(ConnectorMonitorViewer):
    """Viewer for RateConnectorMonitor, visualizing leaky integrate-and-fire connection dynamics."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: RateConnectorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render LI connector monitor using base connector viewer logic."""
        super().render(options=options)
        ref = self.get_ref()
        Zp = self. _get('Zp')
        if Zp is None or len(Zp)==0:
            print('WARN: Zp not found:', self)
            return
        K = filter_count(['Zp', 'Zn'], options, ref)
        if K>0:
            n = Zp[0].shape[0]*Zp[0].shape[1]
            height = n/4
            #print('!n:', n, 'height:', height)
            _,axs = self._axes(K, height=height)
            tt = self._signal_changes()
            k = 0
            def _plot_Cx(ax, k):
                self._plot_input(ax)
                if ref is not None and ref.params is not None:
                    ax.hlines([k], 0, len(Wp), color='r', lw=.5, linestyle= '--')

            if filter('Zp', options, ref):
                self._plot_spikes(['Zp'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1
            if filter('Zn', options, ref):
                self._plot_spikes(['Zn'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1

            #self._plot_mt(['dM'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMp'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMn'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            plt.show()
        return self

class LIConnectorMonitorViewer(ConnectorMonitorViewer):
    """Viewer for LIConnectorMonitor, visualizing leaky integrate-and-fire connection dynamics."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: LIConnectorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render LI connector monitor using base connector viewer logic."""
        super().render(options=options)
        ref = self.get_ref()
        Wp = self. _get('Wp')
        if Wp is None or len(Wp)==0:
            print('WARN: Wp not found:', self)
            return
        K = filter_count(['Wp', 'Wn', 'Zp', 'Zn', '_Cp', '_Cn'], options, ref)
        if K>0:
            n = Wp[0].shape[0]*Wp[0].shape[1]
            height = n/4
            #print('!n:', n, 'height:', height)
            _,axs = self._axes(K, height=height)
            tt = self._signal_changes()
            k = 0
            if filter('Wp', options, ref):
                self._plot_spikes(['Wp'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1
            if filter('Wn', options, ref):
                self._plot_spikes(['Wn'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1

            def _plot_Cx(ax, k):
                self._plot_input(ax)
                if ref is not None and ref.params is not None:
                    ax.hlines([k], 0, len(Wp), color='r', lw=.5, linestyle= '--')

            if filter('_Cp', options, ref):                   
                self._plot_mt(['_Cp'], callback=lambda ax: _plot_Cx(ax, ref.params.k_p), options=options, ax=axs[k])
                k += 1
            if filter('_Cn', options, ref):        
                self._plot_mt(['_Cn'], callback=lambda ax: _plot_Cx(ax, ref.params.k_n), options=options, ax=axs[k])
                k += 1            

            if filter('Zp', options, ref):
                self._plot_spikes(['Zp'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1
            if filter('Zn', options, ref):
                self._plot_spikes(['Zn'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1

            #self._plot_mt(['dM'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMp'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMn'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            plt.show()
        return self
