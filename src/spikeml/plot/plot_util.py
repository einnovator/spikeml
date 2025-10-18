from typing import (
    Any, Callable, Optional, Sequence, Union, Tuple, Dict, List
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CMAP_X: str = 'rainbow'
VMIN: float = 0.0
VMAX: float = 1.0
CMAP_M: str = 'seismic'


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
Callback = Callable[[Axes], None]

def plot_hist(
    data: ArrayLike,
    title: Optional[str] = None,
    bins: Union[int, Sequence[float]] = 10,
    align: str = 'mid',
    ax: Optional[Axes] = None
) -> None:
    """
    Plot a histogram of numerical data.

    Parameters
    ----------
    data : array-like
        Input data to plot as a histogram.
    title : str, optional
        Title of the plot. Default is None.
    bins : int or sequence, optional
        Number of bins or explicit bin edges. Default is 10.
    align : {'left', 'mid', 'right'}, optional
        Alignment of the bins. Default is 'mid'.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    None
        Displays the histogram. Does not return data.
    """

    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]

    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)
    counts, bins, patches = ax.hist(data, bins=bins, align=align)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.set_xticks(bin_centers, [f'{c:.1f}' for c in bin_centers])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    if fig is not None:
        #plt.tight_layout()
        plt.show()

def plot_data(
    data: Union[ArrayLike, Dict[str, ArrayLike], List[ArrayLike]],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    label: Optional[Union[str, List[str]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    _type: Optional[str] = None,
    callback: Optional[Callback] = None,
    ax: Optional[Axes] = None
) -> None:
    """
    Plot one or more data series as a line or stem plot.

    Parameters
    ----------
    data : array-like, dict, or list
        Input data to plot. Can be:
        - A NumPy array of shape (T,) or (T, N)
        - A list of arrays (stacked automatically)
        - A dict mapping labels to arrays
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    label : str or list of str, optional
        Series label(s) for the legend.
    ylim : tuple(float, float), optional
        Y-axis limits (min, max).
    _type : {'stem', 'line', None}, optional
        Type of plot to draw. Default is 'line'; if 'stem', uses `ax.stem()`.
    callback : callable, optional
        Function that receives the `ax` after plotting, for custom modification.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the plot on. If None, creates a new figure.

    Returns
    -------
    None
        Displays the plot.
    """

    if type(data)==list: data = np.stack(data)
    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]

    def _plot(data, label=None):
        if _type=='steam':
            _ = ax.stem(data, label=label)
        else:
            _ = ax.plot(data, label=label)
        
    if isinstance(data, dict):
        for key,data_ in data.items():
            _plot(data_, key)
    else:
        if len(data.shape)==2 and isinstance(label, str):
            label = [ f'{label}[{i}]' for i in range(0, data.shape[1]) ]
        _plot(data, label) 
            
    if callback is not None:
        callback(ax)
    
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)

    if isinstance(data, dict) or label is not None:
        ax.legend(fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=6)
    if ylabel is not None:
        ax.set_xlabel(ylabel, fontsize=6)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if fig is not None:
        #plt.tight_layout()
        plt.show()

def plot_lidata(
    x: ArrayLike,
    level: float,
    level_color: str = 'r',
    level_linestyle: str = '--',
    level_linewidth: float = 1.0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    _type: Optional[str] = None,
    callback: Optional[Callback] = None,
    ax: Optional[Axes] = None
) -> None:
    """
    Plot data with a horizontal reference line at a specified level.

    Parameters
    ----------
    x : array-like
        Data to plot.
    level : float
        Value at which to draw the horizontal reference line.
    level_color : str, optional
        Color of the reference line. Default is 'r' (red).
    level_linestyle : str, optional
        Line style for the reference line. Default is '--'.
    level_linewidth : float, optional
        Line width for the reference line. Default is 1.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    label : str, optional
        Legend label for the plotted data.
    ylim : tuple(float, float), optional
        Y-axis limits. If not set, automatically extends slightly above `level`.
    _type : {'stem', 'line', None}, optional
        Type of data plot.
    callback : callable, optional
        Function that modifies the axis after plotting.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the plot on.

    Returns
    -------
    None
        Displays the plot.
    """

    if type(x)==list: x = np.stack(x)
    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]

    if ylim:
        ylim=[0,level*1.2]
    plot_data(x, title=title, xlabel=xlabel, ylabel=ylabel, label=None, ylim=ylim, _type=_type, callback=callback, ax=ax)
    plt.axhline(y=level, color=level_color, linestyle=level_linestyle, linewidth=level_linewidth)
    if fig is not None:
        #plt.tight_layout()
        plt.show()


def plot_input(
    data: ArrayLike,
    ylim: Optional[Tuple[float, float]] = None,
    color: str = 'b',
    lw: float = 0.5,
    va: str = 'bottom',
    fontsize: int = 5,
    ax: Optional[Axes] = None
) -> None:
    """
    Plot discrete input signal transitions with annotated symbol values.

    Draws vertical dashed lines at time steps where the input signal changes value
    and labels each segment with its symbol.

    Parameters
    ----------
    data : array-like
        Input signal (1D or 2D array) representing symbol or state values over time.
    ylim : tuple(float, float), optional
        Vertical limits of the plot.
    color : str, optional
        Color of vertical lines and text. Default is 'b' (blue).
    lw : float, optional
        Line width of transition markers. Default is 0.5.
    va : {'top', 'bottom', 'center'}, optional
        Vertical alignment of the text annotations. Default is 'bottom'.
    fontsize : int, optional
        Font size for annotations. Default is 5.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the plot. Creates a new figure if None.

    Returns
    -------
    None
        Displays the input signal transitions.
    """
    
    if data is None:
        return
    if type(data)==list: data = np.stack(data)

    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]
        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlim(0, data.shape[0])
        
    dv = np.any(data[1:] != data[:-1], axis=1)
    x = np.where(dv)[0] + 1
    if ylim is None:
        ylim = ax.get_ylim() 
    ymin,ymax =  ylim
    #print(ymin, x)
    s = data[0]
    ax.vlines(x, ymin, ymax, color=color, lw=lw, linestyle='dashed')

    y = ymin + (ymax-ymin)*.01
    ax.text(0, y, f'{s}', ha='left', va=va, color=color, fontsize=fontsize)
    for i in range(0, x.shape[0]):
        j = x[i]
        s = data[j]
        ax.text(j, y, f'{s}', ha='left', va=va, color=color, fontsize=fontsize)

    #for i in range(0,data

    if fig is not None:
        #plt.tight_layout()
        plt.show()


def plot_xt(
    data: ArrayLike,
    vmin: float = VMIN,
    vmax: float = VMAX,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    aspect: Union[str, float] = 'auto'
) -> None:
    """
    Plot a time-vs-feature heatmap (transposed data).

    Parameters
    ----------
    data : array-like
        2D data array of shape (T, N), representing feature values over time.
    vmin, vmax : float, optional
        Minimum and maximum values for the color scale.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    aspect : {'auto', 'equal', float}, optional
        Aspect ratio for the image. Default is 'auto'.

    Returns
    -------
    None
        Displays the heatmap.
    """
    
    
    if type(data)==list: data = np.stack(data)
    data = data.T
    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]
        
    _ = ax.imshow(data, cmap=CMAP_X, vmin=vmin, vmax=vmax, interpolation='nearest', aspect=aspect)
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.tick_params(left=False, bottom=True)
    ax.set_yticklabels([])
    ax.set_anchor('W')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=6)
    #else:
    #    ax.set_xticklabels([])    
    if fig is not None:
        #plt.tight_layout()
        plt.show()

def plot_mt(
    data: np.ndarray,
    title: Optional[str] = None,
    name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_cols: int = -1,
    ax: Optional[Axes] = None,
    aspect: Optional[Union[str, float]] = None,
    callback: Optional[Callback] = None
) -> None:
    """
    Plot multiple time-varying matrices as a set of overlaid line plots.

    Parameters
    ----------
    data : ndarray
        3D array of shape (T, M, N), representing a time series of matrices.
    title : str, optional
        Title of the plot.
    name : str, optional
        Base name used for legend entries (e.g., "A" for "A[i,j]").
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_cols : int, optional
        Number of columns in the legend. If < 0, uses one per matrix. Default is -1.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates a new figure if None.
    aspect : {'auto', 'equal', float}, optional
        Aspect ratio (not used in this function, reserved for consistency).
    callback : callable, optional
        Function called with the axis after plotting for custom styling.

    Returns
    -------
    None
        Displays the plot.
    """

    if type(data)==list: data = np.stack(data)

    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]

    if name is None:
        name = ""
    for j in range(0, data.shape[2]):
        for i in range(0, data.shape[1]):
            data_ = data[:,i,j]
            _ = ax.plot(data_, label=f'{name}[{i},{j}]')

    if callback is not None:
        callback(ax)
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)
    ax.tick_params(left=False, bottom=True)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=6)
    #else:
    #    ax.set_xticklabels([])  
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=6)
    #else:
    #    ax.set_yticklabels([]) 
    if legend_cols<0:
        legend_cols = data.shape[2]
    ax.legend(loc='upper right', fontsize=6, ncols=legend_cols)
    if fig is not None:
        #plt.tight_layout()
        plt.show()

def plot_spikes(
    data: np.ndarray,
    title: Optional[str] = None,
    name: Optional[str] = None,
    xlabel: Optional[str] = None,
    callback: Optional[Callback] = None,
    ax: Optional[Axes] = None
) -> None:
    """
    Visualize spike trains as vertical markers over time.

    Parameters
    ----------
    data : ndarray
        Binary spike data:
        - 2D array of shape (T, N): N neurons, T time steps.
        - 3D array of shape (T, M, N): multiple layers or groups of neurons.
    title : str, optional
        Title of the plot.
    name : str, optional
        Base name for neuron labels. Default is empty string.
    xlabel : str, optional
        Label for the x-axis.
    callback : callable, optional
        Function that modifies the axis after plotting.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Creates a new figure if None.

    Returns
    -------
    None
        Displays the raster plot of spike events.
    """
    
    if type(data)==list: data = np.stack(data)

    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]

    if name is None:
        name = ""
    hpad = 1
    vpad = 5
    height = 10
    #ax.set_xlim(0, data.shape[0])
    #ax.set_ylim(0, (data.shape[1]+1)*(height+vpad))
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()

    yy = []
    if len(data.shape)==3:
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):
                y = ymax-(j+i*data.shape[1]+1)*(height+vpad) 
                yy.append(y)
                indexes = np.nonzero(data[:,i,j])[0]
                x = indexes*hpad
                ax.vlines(x, y, y+height, color='b')
                ax.text(0, y, f'[{i},{j}]', ha='right', fontsize=6)
                ax.text(data.shape[0], y, f'{x.shape[0]} ', ha='left', fontsize=6)
        ax.hlines(yy, 0, data.shape[0], color='b', lw=.5)

    else:
        for i in range(0, data.shape[1]):
            y = ymax-(i+1)*(height+vpad) 
            yy.append(y)
            indexes = np.nonzero(data[:,i])[0]
            x = indexes*hpad
            ax.vlines(x, y, y+height, color='b')
            ax.text(0, y, f'{i} ', ha='right', fontsize=6)
            ax.text(data.shape[0], y, f'{x.shape[0]} ', ha='left', fontsize=6)

        ax.hlines(yy, 0, data.shape[0], color='b', lw=.5)

    if callback is not None:
        callback(ax)
            
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)
    ax.tick_params(left=False, bottom=True)
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=6)
    #else:
    #    ax.set_xticklabels([])    
    ax.margins(x=0.05, y=0.1)
    if fig is not None:
        #plt.tight_layout()
        plt.show()


def imshow_matrix(
    data: np.ndarray,
    title: Optional[str] = None,
    lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    tstyle: Optional[Union[Dict[str, Any], int]] = None,
    cmap: str = CMAP_M,
    aspect: Optional[Union[str, float]] = None,
    interpolation: str = 'nearest',
    ax: Optional[Axes] = None
) -> None:
    """
    Display a single 2D matrix as an image.

    Parameters
    ----------
    data : ndarray
        2D matrix to visualize.
    title : str, optional
        Title of the plot.
    lim : tuple(float, float), optional
        Color scale limits (vmin, vmax). Default is (None, None).
    tstyle : dict or int, optional
        Style dictionary or font size for the title text.
    cmap : str, optional
        Colormap name. Default is 'seismic'.
    aspect : {'auto', 'equal', float}, optional
        Aspect ratio of the image. Default is None.
    interpolation : str, optional
        Interpolation method for imshow. Default is 'nearest'.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates a new one if None.

    Returns
    -------
    None
        Displays the image.
    """

    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(1*ncols, 1*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]
    if lim==None:
        lim = (None, None)

    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
        
    ax.imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation=interpolation, aspect=aspect)
    ax.axis('off')
    if title is not None:
        if tstyle is None:
            tstyle = { 'fontsize': 8 }
        elif isinstance(tstyle, int):
            tstyle = { 'fontsize': tstyle }
        ax.set_title(title, **tstyle)
        
    if fig is not None:
        #plt.tight_layout()
        plt.show()

def imshow_nmatrix(
    data: np.ndarray,
    tt: Optional[List[int]] = None,
    tstep: int = 1,
    tk: Optional[int] = None,
    ncols: int = 8,
    title: Optional[str] = None,
    tstyle: Optional[Union[Dict[str, Any], int]] = None,
    lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    cmap: str = CMAP_M,
    aspect: Optional[Union[str, float]] = None,
    interpolation: str = 'nearest',
    axs: Optional[np.ndarray] = None
) -> None:
    """
    Display a sequence of matrices as a grid of images over time.

    Parameters
    ----------
    data : ndarray
        3D array (T, H, W) representing a time series of matrices.
    tt : list[int], optional
        Explicit list of time indices to plot. If None, computed automatically.
    tstep : int, optional
        Step between frames when sampling from `data`. Default is 1.
    tk : int, optional
        Number of frames to plot (used if `tt` is None).
    ncols : int, optional
        Number of columns in the subplot grid. Default is 8.
    title : str, optional
        Base title prefix for each subplot.
    tstyle : dict or int, optional
        Title style or font size.
    lim : tuple(float, float), optional
        Color scale limits for all subplots.
    cmap : str, optional
        Colormap name. Default is 'seismic'.
    aspect : {'auto', 'equal', float}, optional
        Aspect ratio of each subplot image.
    interpolation : str, optional
        Interpolation mode for imshow.
    axs : ndarray of Axes, optional
        Preexisting axes grid to draw on.

    Returns
    -------
    None
        Displays the grid of images representing temporal evolution.
    """

    fig = None
    if tt is None:
        if tk is not None:
            tstep = len(data)/tk
            a = np.linspace(0, len(data), num=tk)
            tt = [int(a[i]) for i in range(0, a.shape[0])] 
        else:
            tt = [t for t in range(0, len(data), tstep)]   
        #print(tk, tstep, tt)
        tt[-1] = len(data)-1
    if axs is None:
        nrows = int(len(tt) / ncols) + (1 if len(tt)%ncols!=0 else 0)
        ncols = min(ncols, len(tt))
        fig, axs = plt.subplots(nrows, ncols, figsize=(1*ncols, 1*nrows))

    i = j = 0
    for t in tt:    
        title_ = f'{title} ({t})' if title is not None else f'({t})'
        ax = axs if nrows==1 and ncols==1 else axs[j] if nrows==1 else axs[i][j]
        imshow_matrix(data[t], title=title_, lim=lim, tstyle=tstyle, cmap=cmap, aspect=aspect, interpolation=interpolation, ax=ax)
        j += 1
        if j==ncols:
            j = 0
            i += 1
    if nrows>1:
        while j<ncols:
            ax = axs[i][j]
            ax.tick_params(left=False, bottom=False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')
            j += 1
        
    if fig is not None:
        #plt.tight_layout()
        plt.show()
        
def _plot_ref(ref: Any) -> None:
    """
    Plot all monitored signals and matrices from a reference simulation object.

    Automatically detects available fields in `ref.monitor` (e.g., `x`, `s`, `y`, `A`, `B`, ...)
    and visualizes each using appropriate plotting functions.

    Parameters
    ----------
    ref : object
        Reference object with attributes:
        - `name`: identifier for labeling plots.
        - `monitor`: single monitor or list of monitors containing recorded data.
        - Optional fields: `refs`, `vmin`, `vmax`.

    Returns
    -------
    None
        Generates and displays multiple plots summarizing recorded data.
    """

    def _plot_monitor(monitor, ):
        if hasattr(monitor, 'x') and monitor.x is not None:
            plot_xt(monitor.x, vmin=ref.vmin, vmax=ref.vmax, title=f'{ref.name}.x(t)')
        if hasattr(monitor, 's') and monitor.s is not None:
            plot_xt(monitor.s, vmin=ref.vmin, vmax=ref.vmax, title=f'{ref.name}.s(t)')
        if hasattr(monitor, 'y') and monitor.y is not None:
            plot_xt(monitor.y, vmin=ref.vmin, vmax=ref.vmax, title=f'{ref.name}.y(t)')
        if hasattr(monitor, 'A') and monitor.A is not None:
            plot_mt(monitor.A, title=f'{ref.name}.A(t)', name='A')
        if hasattr(monitor, 'B') and monitor.B is not None:
            plot_mt(monitor.B, title=f'{ref.name}.A(t)', name='B')
        if hasattr(monitor, 'C') and monitor.C is not None:
            plot_mt(monitor.C, title=f'{ref.name}.C(t)', name='C')
        if hasattr(monitor, 'D') and monitor.D is not None:
            plot_mt(monitor.D, title=f'{ref.name}.D(t)', name='D')
        
    if hasattr(ref, 'refs'):
        for ref_ in ref.refs:
           plot_ref(ref_)
    else:
        if hasattr(ref, 'monitor') and ref.monitor is not None:
            if isinstance(ref.monitor, list):
                for monitor in ref.monitor:
                    _plot_monitor(monitor, ref)
            else:
                    _plot_monitor(monitor, ref)

