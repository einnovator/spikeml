from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from spikeml.core.monitor import Monitor

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

class MonitorViewer():
    """Base class for viewing Monitor data."""

    def __init__(self, monitor: Monitor) -> None:
        """
        Args:
            monitor: Monitor object to visualize.
        """
        #super().__init__()
        self.monitor = monitor

    def _get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve stored data from monitor."""
        monitor = self.monitor
        if hasattr(monitor, key):
            data = getattr(monitor, key)
            if data is not None and len(data)>0:
                return data
        return None

    def _axes(self, nrows: int = 1, ncols: int = 1, width: int = 10, height: int = 1):
        """Create matplotlib axes."""
        fig, axs = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows))
        return fig, axs
        
    def _prefix(self, prefix: Optional[str] = None) -> str:
        """Return monitor prefix string."""
        prefix = self.monitor._prefix(prefix)
        if prefix is None:
            prefix = ''
        else:
            prefix = f'{prefix}.'
        return prefix

    def _include(self, key: str, options: Optional[Union[str, List[str], dict]] = None) -> bool:
        """Check if a key should be included based on options."""
        if options is None:
            return True
        if isinstance(options, str):
            options = options.strip()
            if len(options)==0: 
                options = None
            else:
                options = [tag.strip() for tag in options.split(',')]
        if options is not None:
            if isinstance(options, list) and not key in options:
                return False
            if isinstance(options, dict):
                if key in options and options[key]==False:
                    return False
        return True

    def _render(self, renderer, keys, title=None, shared=False, options=None):        
        prefix = self._prefix()
        if not isinstance(keys, list):
            keys = [keys]
        if isinstance(options, str):
            options = options.strip()
            if len(options)==0: 
                options = None
            else:
                options = [tag.strip() for tag in options.split(',')]
        if shared:
            data= {}
            for key in keys:
                if not self._include(key, options):
                    continue 
                data_ = self._get(key)
                if data_ is not None:
                    data[key] = data_
            if len(data)>0:
                if title is None:
                    title = ','.join(keys)
                renderer(data, key, f'{prefix}{title}', 0)
        else:
            index = 0
            for key in keys:
                if options is not None:
                    if not self._include(key, options):
                        continue 
                data = self._get(key)
                if data is not None:
                    renderer(data, key, f'{prefix}{key}', index)
                    index += 1
                #else:
                #    print(f'WARN: no data: {key}')
                    
    def _ax(self, axs, i, j=0):
        if axs is None:
            return None
        ax = axs if isinstance(axs, mplt.axes.Axes) else axs[i] if len(axs.shape)==1 else axs[i][j]
        return ax
    
    def _plot_xt(self, keys, options=None, ax=None):
        self._render(lambda data, key, title, index: plot_xt(data, title=title, aspect='auto', ax=self._ax(ax, index)), keys, options=options)

    def _plot_data(self, keys, _type=None, ylim=None, shared=False, callback=None, options=None, ax=None):
        self._render(lambda data, key, title, index: plot_data(data, title=title, label=key, _type=_type, ylim=ylim, callback=callback, ax=self._ax(ax, index)), keys, shared=shared, options=options)

    def _plot_lidata(self, keys, level, _type=None, ylim=None, shared=False, callback=None, options=None, ax=None):
        self._render(lambda data, key, title, index: plot_lidata(data, level, title=title, label=key, _type=_type, ylim=ylim, callback=callback, ax=self._ax(ax, index)), keys, shared=shared, options=options)
        
    def _plot_mt(self, keys, callback=None, options=None, ax=None):
        self._render(lambda data, key, title, index: plot_mt(data, title=title, callback=callback, ax=self._ax(ax, index)), keys, options=options)

    def _plot_spikes(self, keys, callback=None, options=None, ax=None):
        self._render(lambda data, key, title, index: plot_spikes(data, title=title, callback=callback, ax=self._ax(ax, index)), keys, options=options)

    def _imshow_nmatrix(self, keys, tk=10, ncols=10, options=None, ax=None):
        self._render(lambda data, key, title, index: imshow_nmatrix(data, title=title, tk=tk, ncols=ncols), keys, options=options)

    def render(self, options=None):
        pass

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
