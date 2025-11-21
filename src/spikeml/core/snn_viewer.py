
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
        from spikeml.core.snn import SSensor
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

class LayerMonitorViewer(SensingMonitorViewer):
    """Viewer for LayerMonitor, visualizing spikes, membrane potential, and outputs."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: LayerMonitor object to visualize.
        """
        super().__init__(monitor)


    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render monitor data including spikes, membrane potentials, outputs."""
        tt = self._signal_changes()
        ref = self.get_ref()
        K = filter_count(['zy', 'y', 'y'], options, ref)
        if K>0:
            _,axs = self._axes(K)        
            k = 0 
            if filter('zy', options, ref):
                self._plot_spikes('zy', tt=tt, callback=lambda ax: self._plot_input(va='top', ax=ax), options=options, ax=axs[0])
                k +=1
            if filter('y', options, ref):
                self._plot_xt(['y'], options=options, ax=axs[k+0])
                self._plot_data(['y'], options=options, ax=axs[k+1], callback=lambda ax: self._plot_input(ax))
            plt.show()
        if filter('u', options, ref):
            self._plot_data(['u', 'us'], shared=True, callback=lambda ax: self._plot_input(ax), options=options)

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
        K = filter_count(['zy', 'y', 'y', 'r'], options, ref)
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
            if filter('r', options, ref):
                self._plot_data(['r'], options=options, ax=axs[k], callback=lambda ax: self._plot_input(ax))
                k += 1
            plt.show()
        if filter('u', options, ref):
            self._plot_data(['u', 'us'], shared=True, callback=lambda ax: self._plot_input(ax), options=options)

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
            if filter('Zp', options, ref):
                self._plot_spikes(['Zp'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1
            if filter('Zn', options, ref):
                self._plot_spikes(['Zn'], tt=tt, callback=lambda ax: self._plot_input(ax), options=options, ax=axs[k])
                k += 1

            def _plot_Cx(ax):
                self._plot_input(ax)
                if ref is not None and ref.params is not None:
                    ax.hlines([ref.params.c_k], 0, len(Wp), color='r', lw=.5, linestyle= '--')

            if filter('_Cp', options, ref):                   
                self._plot_mt(['_Cp'], callback=_plot_Cx, options=options, ax=axs[k])
                k += 1
            if filter('_Cn', options, ref):        
                self._plot_mt(['_Cn'], callback=_plot_Cx, options=options, ax=axs[k])
                k += 1            
            #self._plot_mt(['dM'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMp'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            #self._plot_mt(['dMn'], callback=lambda ax: self._plot_input(ax), options=options, ax=axs[0])
            plt.show()
        return self
    
class ErrorMonitorViewer(MonitorViewer):
    """Viewer for ErrorMonitor, visualizing error metrics over time."""

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(monitor)

    def render(self, options: Optional[Union[dict, List[str], str]] = None) -> None:
        """Render error metrics and smoothed signals."""
        monitor = self.monitor
        ref = self.get_ref()
        K = filter_count(['err', 'sm'], options, ref)
        if K>0:
            _,axs = self._axes(K)    
            k = 0 
            if filter('err', options, ref):                   
                self._plot_data(['err', 'merr'], shared=True, ylim=(0,1.1), options=options, ax=axs[k])
                k += 1
            if filter('sm', options, ref):
                self._plot_data(['sm'], options=options, ax=axs[k])
                k += 1
            plt.show()
