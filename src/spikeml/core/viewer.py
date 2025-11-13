from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from spikeml.core.monitor import Monitor

import matplotlib.pyplot as plt
import matplotlib as mplt

from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix

class MonitorViewer():
    """Base class for viewing data stored in a Monitor object."""

    def __init__(self, monitor: "Monitor") -> None:
        """
        Initialize a monitor viewer.

        Args:
            monitor (Monitor): The monitor instance whose data will be visualized.
        """
        #super().__init__()
        self.monitor = monitor

    def _get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve stored data from the monitor by key.

        Args:
            key (str): Attribute name to retrieve from the monitor.

        Returns:
            Optional[np.ndarray]: Stored data if available, otherwise None.
        """
        monitor = self.monitor
        if hasattr(monitor, key):
            data = getattr(monitor, key)
            if data is not None and len(data)>0:
                return data
        return None

    def _axes(
        self,
        nrows: int = 1,
        ncols: int = 1,
        width: int = 10,
        height: int = 1
    ) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
        """
        Create a matplotlib figure and axes grid.

        Args:
            nrows (int, optional): Number of subplot rows. Defaults to 1.
            ncols (int, optional): Number of subplot columns. Defaults to 1.
            width (int, optional): Width of each subplot. Defaults to 10.
            height (int, optional): Height of each subplot. Defaults to 1.

        Returns:
            Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]: Figure and axes handles.
        """
        fig, axs = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows))
        plt.subplots_adjust(hspace=1)
        return fig, axs
        
    def _prefix(self, prefix: Optional[str] = None) -> str:
        """
        Generate a name prefix for titles and labels.

        Args:
            prefix (Optional[str], optional): Optional override prefix. Defaults to None.

        Returns:
            str: Formatted prefix string.
        """
        prefix = self.monitor._prefix(prefix)
        if prefix is None:
            prefix = ''
        else:
            prefix = f'{prefix}.'
        return prefix

    def _include(
        self,
        key: str,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None
    ) -> bool:
        """
        Check if a given key should be plotted based on provided options.

        Args:
            key (str): The data key to check.
            options (Optional[Union[str, List[str], Dict[str, Any]]], optional):
                Filtering or selection options (e.g., list of allowed keys). Defaults to None.

        Returns:
            bool: True if the key should be included, False otherwise.
        """
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

    def _render(
        self,
        renderer: Callable[[np.ndarray, str, str, int], None],
        keys: Union[str, List[str]],
        title: Optional[str] = None,
        shared: bool = False,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None
    ) -> None:
        """
        Central rendering loop for plotting monitor data.

        Args:
            renderer (Callable): Plotting function taking (data, key, title, index).
            keys (Union[str, List[str]]): Keys to visualize.
            title (Optional[str], optional): Plot title. Defaults to None.
            shared (bool, optional): If True, combines multiple signals in one plot. Defaults to False.
            options (Optional[Union[str, List[str], Dict[str, Any]]], optional): Filter options. Defaults to None.
        """      
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
                    
    def _ax(
        self,
        axs: Optional[Union[plt.Axes, np.ndarray]],
        i: int,
        j: int = 0
    ) -> Optional[plt.Axes]:
        """
        Select an axis from an axes array or return a single axis.

        Args:
            axs (Optional[Union[plt.Axes, np.ndarray]]): Axes object(s).
            i (int): Row index.
            j (int, optional): Column index. Defaults to 0.

        Returns:
            Optional[plt.Axes]: Selected axis or None.
        """
        if axs is None:
            return None
        ax = axs if isinstance(axs, mplt.axes.Axes) else axs[i] if len(axs.shape)==1 else axs[i][j]
        return ax
    
    def _plot_xt(
        self,
        keys: Union[str, List[str]],
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot temporal data (e.g., spike traces)."""
        self._render(lambda data, key, title, index: plot_xt(data, title=title, aspect='auto', ax=self._ax(ax, index)), keys, options=options)

    def _plot_data(
        self,
        keys: Union[str, List[str]],
        _type: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        shared: bool = False,
        callback: Optional[Callable[[plt.Axes], None]] = None,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot general data values from the monitor."""        
        self._render(lambda data, key, title, index: plot_data(data, title=title, label=key, _type=_type, ylim=ylim, callback=callback, ax=self._ax(ax, index)), keys, shared=shared, options=options)

    def _plot_lidata(
        self,
        keys: Union[str, List[str]],
        level: float,
        _type: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        shared: bool = False,
        callback: Optional[Callable[[plt.Axes], None]] = None,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot leaky-integrator data."""
        self._render(lambda data, key, title, index: plot_lidata(data, level, title=title, label=key, _type=_type, ylim=ylim, callback=callback, ax=self._ax(ax, index)), keys, shared=shared, options=options)
        
    def _plot_mt(
        self,
        keys: Union[str, List[str]],
        callback: Optional[Callable[[plt.Axes], None]] = None,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot matrix data."""
        self._render(lambda data, key, title, index: plot_mt(data, title=title, callback=callback, ax=self._ax(ax, index)), keys, options=options)

    def _plot_spikes(
        self,
        keys: Union[str, List[str]],
        tt: Optional[Any] = None, 
        callback: Optional[Callable[[plt.Axes], None]] = None,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot spike raster data."""
        self._render(lambda data, key, title, index: plot_spikes(data, tt=tt, title=title, callback=callback, ax=self._ax(ax, index)), keys, options=options)

    def _imshow_nmatrix(
        self,
        keys: Union[str, List[str]],
        tk: int = 10,
        ncols: int = 10,
        options: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Render normalized matrices as images."""
        self._render(lambda data, key, title, index: imshow_nmatrix(data, title=title, tk=tk, ncols=ncols), keys, options=options)

    def render(self, options: Optional[Any] = None) -> None:
        """Base render method (to be overridden by subclasses)."""
        pass
