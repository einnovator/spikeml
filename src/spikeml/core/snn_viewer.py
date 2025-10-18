
from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer

from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix

class SSensorMonitorViewer(MonitorViewer):
    """Viewer for SSensorMonitor, visualizing sensor spikes and input."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: SSensorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render the SSensor monitor data including spikes and sensor input."""
        def _plot_input(ax):
            plot_input(self._get('sx'), ax=ax)
        if self._get('sx') is not None:
            _,axs = self._axes(2)
            self._plot_xt(['sx'], options=options, ax=axs[0])
            self._plot_data(['sx'], callback=_plot_input, options=options, ax=axs[1]) 
            plt.show()
        _,axs = self._axes(3)
        self._plot_xt(['s'], options=options, ax=axs[0])
        self._plot_data(['s'], callback=_plot_input, options=options, ax=axs[1]) 
        self._plot_spikes('zs', callback=lambda ax: plot_input(self._get('sx'), va='top', ax=ax), options=options, ax=axs[2])
        plt.show()

class SNNMonitorViewer(MonitorViewer):
    """Viewer for SNNMonitor, visualizing spikes, membrane potential, and outputs."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: SNNMonitor object to visualize.
        """
        super().__init__(monitor)

    def _get_sensor_input(self) -> Optional[np.ndarray]:
        """Retrieve sensor input 'sx' from connected SSensor layer."""
        sensor = self.monitor.ref._parent.find(SSensor)
        if sensor is not None and sensor.monitor is not None:
            return sensor.monitor._get('sx')
        return None

    def _plot_input(self, ax: plt.Axes, va: Optional[str] = None) -> None:
        """Plot sensor input on the provided axis."""
        sx = self._get_sensor_input()
        if sx is not None:
            plot_input(sx, va=va, ax=ax)


    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render SNN monitor data including spikes, membrane potentials, outputs."""
        _,axs = self._axes(3)
        self._plot_xt(['s'], options=options, ax=axs[0])
        self._plot_data(['s'], callback=_plot_input, options=options, ax=axs[1]) 
        self._plot_spikes('zs', callback=lambda ax: self._plot_input(ax=ax, va='top'), options=options, ax=axs[2])
        plt.show()
        _,axs = self._axes(2)        
        self._plot_xt(['x'], options=options, ax=axs[0])
        self._plot_lidata('x', self.monitor.ref.params.k_x, options=options, ax=axs[1])
        plt.show()
        _,axs = self._axes(4)        
        self._plot_spikes('zy', callback=lambda ax: self.plot_input(va='top', ax=ax), options=options, ax=axs[0])
        self._plot_xt(['y'], options=options, ax=axs[1])
        self._plot_data(['y'], options=options, ax=axs[2], callback=lambda ax: _plot_input(ax))
        self._plot_data(['r'], options=options, ax=axs[3], callback=lambda ax: _plot_input(ax))
        plt.show()
        _,axs = self._axes(2)        
        self._plot_xt(['k_x'], options=options, ax=axs[0])
        self._plot_data(['k_x'], options=options, ax=axs[1], callback=lambda ax: _plot_input(ax))
        plt.show()
        self._plot_data(['u', 'us'], shared=True, callback=_plot_input, options=options)

class SSNNMonitorViewer(SNNMonitorViewer):
    """Viewer for SSNNMonitor, visualizing stochastic spiking layers."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: SSNNMonitor object to visualize.
        """
        super().__init__(monitor)

        
    def render(self, options=None):

        self._plot_spikes('zy', callback=lambda ax: self._plot_input(ax), options=options)
        self._plot_xt(['y'], options=options)
        self._plot_data(['y'], options=options, callback=lambda ax: _plot_input(ax))
        self._plot_data(['u', 'us'], shared=True, callback=lambda ax: _plot_input(ax), options=options)

class ConnectorMonitorViewer(MonitorViewer):
    """Viewer for ConnectorMonitor, visualizing weight matrices and updates."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: ConnectorMonitor object to visualize.
        """
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render SSNN monitor data including spikes, outputs, and aggregated signals."""
        def _plot_input(ax):
            plot_input(self._get('sx'), ax=ax)
        _,axs = self._axes(2)
        self._plot_mt(['M'], callback=_plot_input, options=options, ax=axs[0])
        self._plot_data(['dw'], callback=_plot_input, options=options, ax=axs[1])
        plt.show()
        self._imshow_nmatrix(['M'], options=options)
        
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

class ErrorMonitorViewer(MonitorViewer):
    """Viewer for ErrorMonitor, visualizing error metrics over time."""

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render error metrics and smoothed signals."""
        monitor = self.monitor
        _,axs = self._axes(2)     
        self._plot_data(['err', 'merr'], shared=True, ylim=(0,1.1), options=options, ax=axs[0])
        self._plot_data(['sm'], options=options, ax=axs[1])
        plt.show()
