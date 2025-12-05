import numpy as np
import math
from typing import Optional, Tuple, Union, Any, Dict
from enum import Enum, auto

from spikeml.core.params import Params, NNParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams
from spikeml.core.base import Adapter, Module
from spikeml.core.env import Source
from spikeml.core.snn_monitor import SSensorMonitor, LayerMonitor, SNNMonitor, SSNNMonitor, ConnectorMonitor, LIConnectorMonitor
from spikeml.core.snn_monitor import  ErrorMonitor
from spikeml.core.snn_viewer import  ErrorMonitorViewer

def compute_error(
    s: np.ndarray,
    y: np.ndarray,
    params: Optional['SSNNParams'] = None,
    mean: bool = True,
    strict: bool = False
) -> float:
    """
    Compute an element-wise error between a target signal `s` and a predicted
    output `y`, optionally returning the mean error.

    This error function measures mismatches between the desired signal and the
    model output using asymmetric penalties. Let ``vmax`` denote the maximum
    allowable signal value (taken from `params`):

    Non-strict mode (default)
    -------------------------
    The error penalizes false positives only:
        error_i = (vmax - s_i) * y_i

    Strict mode
    -----------
    The error penalizes both false positives and false negatives:
        error_i = s_i * (vmax - y_i) + (vmax - s_i) * y_i

    In binary cases (s, y ∈ {0, vmax}), this reduces to:

        strict = False:
            s  y  err
            0  0   0
            0  1   1
            1  0   0
            1  1   0

        strict = True:
            s  y  err
            0  0   0
            0  1   1
            1  0   1
            1  1   0

    Parameters
    ----------
    s : np.ndarray
        Target signal. May be binary or continuous in the range [vmin, vmax].
    y : np.ndarray
        Predicted signal. Must have the same shape as `s`.
    params : SSNNParams, optional
        Parameters providing `vmax` and `vmin`. If None, a default instance is used.
    mean : bool, optional
        If True (default), return the mean error value. If False, return the
        full error array.
    strict : bool, optional
        Whether to use the strict variant of the error function.

    Returns
    -------
    float or np.ndarray
        Mean error (if ``mean=True``) or element-wise error array otherwise.

    Examples
    --------
    >>> compute_error(np.array([1, 0]), np.array([1, 1]), strict=False)
    0.5

    >>> compute_error(np.array([1, 0]), np.array([1, 1]), strict=True)
    1.0
    """

    
    if params is None:
        params = SSNNParams()
    if strict:
        p = s * (params.vmax-y)+(params.vmax-s)*y
    else:
        p = (params.vmax-s)*y
    err = p.mean(axis=-1) if mean else p
    #print(f's: {s} ; y: {y}', f'-> err: {err:.2f}')
    return err


class OutputAggregation(Enum):
    DP = auto()
    MEAN = auto()
    SUM = auto()
    SUM_CLIP = auto()
    MAX = auto()
    
def xcompute_error(
    s: np.ndarray,
    y: np.ndarray,
    R: int = 1,
    method: str = 'sum+clip',
    mean: bool = True,
    params: Optional['SSNNParams'] = None,
    strict: bool = False
) -> float:
    """
    Compute an error between `s` and `y` with optional temporal downsampling
    or aggregation prior to evaluation. The final error is computed using
    `compute_error()`.

    Downsampling/Aggregation
    ------------------------
    When `R > 1`, the predicted signal `y` is grouped in blocks of length `R`
    and aggregated using the method specified by `method`:

    - 'dp':
        Repeat (duplicate) each value in `s` to match the length of `y`.
    - 'mean':
        y_block = mean of each block of size R.
    - 'sum':
        y_block = sum over blocks, then clipped into [vmin, vmax].
    - 'sum+clip' (default):
        Same as 'sum'.
    - 'max':
        y_block = max value in each block.

    After aggregation, the function calls:

        compute_error(s, y_aggregated, strict=strict)

    Parameters
    ----------
    s : np.ndarray
        Target signal. If method='dp' and lengths mismatch, it is repeated.
    y : np.ndarray
        Predicted signal.
    R : int, optional
        Downsampling factor (block size). Defaults to 1 (no aggregation).
    method : {'dp', 'mean', 'sum', 'sum+clip', 'max'}, optional
        Aggregation method for `y`. Defaults to 'sum+clip'.
    params : SSNNParams, optional
        Provides `vmin` and `vmax`. Defaults to None.
    strict : bool, optional
        Passed through to `compute_error()`.

    Returns
    -------
    float
        Aggregated error between `s` and (optionally downsampled) `y`.

    Examples
    --------
    >>> s = np.array([0, 1])
    >>> y = np.array([0, 1, 1, 1])
    >>> xcompute_error(s, y, R=2, method='mean')
    compute_error([0, 1], [0.5, 1.0])

    >>> xcompute_error(s, y, R=2, method='max')
    compute_error([0, 1], [1, 1])
    """
    
    if params is None:
        params = SSNNParams()
    if R>1:
        # --- Aggregation methods ---
        if method == OutputAggregation.DP:
            # Downsample prediction by duplicating s
            if s.shape[0] != y.shape[0]:
                s = np.repeat(s, R)
        else:
            # reshape y into blocks
            y = y.reshape(y.shape[0] // R, R)
            if method == OutputAggregation.MEAN:
                y = y.mean(axis=1)
            elif method == OutputAggregation.SUM:
                y = y.sum(axis=1)
            elif method == OutputAggregation.SUM_CLIP:
                y = y.sum(axis=1)
                y = np.clip(y, params.vmin, params.vmax)
            elif method == OutputAggregation.MAX:
                y = y.max(axis=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            
    err = compute_error(s, y, params=params, mean=mean, strict=strict)
    return err

def compute_sg(err, params):
    """Compute error based gain.

    Args:
        err (float): the error_
        params (_type_): parameters

    Returns:
        float: gain
    """
    sg = np.exp(-err*params.e_err)
    return sg


class FeedbackAdapter(Adapter):
    """
    FeedbackAdapter class.
    
    Attributes:
        ref: Referenced Module
        source: the signal Source
        feedback: flag indicating if feedback is enabled (Default: True)
        name: Optional name of the module.
        params: Parameters associated with the module.
        auto_sample: Whether to automatically sample during updates.
        monitor: Optional monitor object for logging or visualization.
        viewer: Optional viewer object for visualization.
        callback: Optional callback function.
    """

    def __init__(self,
                 ref: Optional[Module] = None, 
                 source: Optional[Source] = None,
                 feedback: Optional[bool] = True,
                 name: Optional[str] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = True,
                 monitor: Optional[Any] = True,
                 viewer: Optional[Any] = True,
                 callback: Optional[Any] = None):
        super().__init__(ref, name=name, params=params, auto_sample=auto_sample, callback=callback)        
        self.source = source
        self.feedback = feedback
        self.y, self.zy = None, None
        self.gain = 1
        if feedback:
            if monitor==True:
                monitor = ErrorMonitor(ref=ref)
            if viewer==True:
                viewer = ErrorMonitorViewer(monitor)
        else:
            if isinstance(monitor, bool):
                monitor = None
                viewer = None
        self.monitor=monitor
        self.viewer=viewer


    def propagate(self, sx: Optional[Source] = None, context: Optional[Any]=None) -> Any:
        """
        Compute module output for a given input signal.

        This method should be overridden by subclasses.

        Parameters
        ----------
        s : any
            Input signal or state.

        Returns
        -------
        any
            Output result (default: None).
        """        
        if sx is None:
            sx = self.source.next()
            if sx is None:
                return None

        s = sx
        if self.feedback and self.zy is not None:
            sy = self.params.g*self.zy #self.y
            s = sx + sy
            s *= self.gain
            s = np.clip(s, self.params.vmin, self.params.vmax)
        
        s_ = (s, sx)
        y_ = self.ref(s_, context)
        y,zy = y_ if isinstance(y_, tuple) else (y_, y_)
        self.y,self.zy = y, zy 
        
        self.gain = 1
        if self.feedback:
            #error = compute_error(sx, y)
            #error = xcompute_error(sx, y, R=R, method='sum+clip')
            error = compute_error(sx, zy)
            self.gain = compute_sg(error, self.params)
            if self.auto_sample and self.monitor is not None and context is not None:
                context.set_attr('sx', sx)
                context.set_attr('error', error)
                context.set_attr('gain', self.gain)
                self.monitor.sample(context)
            
        return (s, sx)

class PhasedFeedbackAdapter(Adapter):
    """
    PhasedFeedbackAdapter class.
    
    Attributes:
        ref: Referenced Module
        source: the signal Source
        feedback: flag indicating if feedback is enabled (Default: True)
        name: Optional name of the module.
        params: Parameters associated with the module.
        auto_sample: Whether to automatically sample during updates.
        monitor: Optional monitor object for logging or visualization.
        viewer: Optional viewer object for visualization.
        callback: Optional callback function.
    """

    def __init__(self,
                 ref: Optional[Module] = None, 
                 source: Optional[Source] = None,
                 phases: Optional[list[Any]] = None,
                 feedback: Optional[bool] = True,
                 name: Optional[str] = None,
                 params: Optional[Any] = None,
                 auto_sample: bool = True,
                 monitor: Optional[Any] = True,
                 viewer: Optional[Any] = True,
                 callback: Optional[Any] = None):
        super().__init__(ref, name=name, params=params, auto_sample=auto_sample, callback=callback)        
        self.source = source
        self.feedback = feedback
        self.phases = phases
        self._phases = _phases
        if phases is not None and isinstance(phases, int):
            self._phases = range(0, phases)
        
        self.y, self.zy = None, None
        self.gain = 1
        if feedback:
            if monitor==True:
                monitor = ErrorMonitor(ref=ref)
            if viewer==True:
                viewer = ErrorMonitorViewer(monitor)
        else:
            if isinstance(monitor, bool):
                monitor = None
                viewer = None
        self.monitor=monitor
        self.viewer=viewer


    def propagate(self, sx: Optional[Source] = None, context: Optional[Any]=None) -> Any:
        """
        Compute module output for a given input signal.

        This method should be overridden by subclasses.

        Parameters
        ----------
        s : any
            Input signal or state.

        Returns
        -------
        any
            Output result (default: None).
        """        
        if sx is None:
            sx = self.source.next()
            if sx is None:
                return None

        s = sx
        if self.feedback and self.zy is not None:
            sy = self.params.g*self.zy #self.y
            s = sx + sy
            s *= self.gain
            s = np.clip(s, self.params.vmin, self.params.vmax)
        
        s_ = (s, sx)
        if self.phases is None:
            y_ = self.ref(s_, context)
        else:
            y_ = None
            for phase in self._phases:
                context.phase = phase
                y__ = self.ref(s_, context)
                if y_ is None:
                    y_ = y__

        if y_ is not None:                
            y,zy = y_ if isinstance(y_, tuple) else (y_, y_)
            self.y,self.zy = y, zy 
            
            self.gain = 1
            if self.feedback:
                #error = compute_error(sx, y)
                #error = xcompute_error(sx, y, R=R, method='sum+clip')
                error = compute_error(sx, zy)
                self.gain = compute_sg(error, self.params)
                if self.auto_sample and self.monitor is not None and context is not None:
                    context.set_attr('sx', sx)
                    context.set_attr('error', error)
                    context.set_attr('gain', self.gain)
                    self.monitor.sample(context)
            
        return (s, sx)
