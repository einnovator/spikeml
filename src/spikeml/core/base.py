from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


Callback = Callable[["Component", Any, Any], None]
MonitorType = Union[Any, Sequence[Any]]
ViewerType = Union[Any, Sequence[Any]]
RefsType = Optional[List["Module"]]


class Component():
    """
    Base class representing a functional system component with monitoring,
    visualization, and callback hooks.

    Provides common mechanisms for:
      - Sampling monitored values (`sample`)
      - Rendering visual output (`render`)
      - Executing callbacks after updates (`post_step`)

    Parameters
    ----------
    name : str, optional
        Name identifier for the component.
    params : dict, optional
        Dictionary of configuration parameters.
    auto_sample : bool, optional
        If True, sampling and post-step callbacks are triggered automatically
        during computation steps. Default is False.
    monitor : object or list, optional
        Monitoring object(s) providing a `.sample()` method.
    viewer : object or list, optional
        Viewer object(s) providing a `.render(options)` method.
    callback : callable or list of callables, optional
        Function(s) invoked after each computation step via `post_step(self, s, y)`.

    Attributes
    ----------
    name : str or None
        Name of the component.
    params : dict or None
        Component parameters.
    monitor : object or list or None
        Monitoring interface(s).
    viewer : object or list or None
        Visualization interface(s).
    callback : callable or list or None
        Post-step callback(s).
    auto_sample : bool
        Whether automatic sampling is enabled.
    """    
    def __init__(
        self,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        auto_sample: bool = False,
        monitor: Optional[MonitorType] = None,
        viewer: Optional[ViewerType] = None,
        callback: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        self.name = name
        self.params = params
        self.callback = callback
        self.auto_sample = auto_sample
        self.monitor = monitor
        self.viewer = viewer
    
    def post_step(self, s: Any, y: Any) -> None:
        """
        Invoke registered callbacks after a computation step.

        Parameters
        ----------
        s : any
            Input signal or state for the current step.
        y : any
            Output signal or state from the current step.
        """
        if self.callback is not None:
            if isinstance(self.callback, list):
                for callback in self.callback:
                    callback(self, s, y)
            else:
                self.callback(self, s, y)
    
    def sample(self) -> None:
        """
        Trigger sampling in associated monitor(s).
        """        
        if self.monitor is not None:
            if isinstance(self.monitor, list):
                for monitor in self.monitor:
                    monitor.sample()
            else:
                self.monitor.sample()

    def render(self, options: Any) -> None:
        """
        Render component visualization through associated viewer(s).

        Parameters
        ----------
        options : dict or any
            Rendering configuration or options passed to viewers.
        """        
        if self.viewer is not None:
            if isinstance(self.viewer, list):
                for viewer in self.viewer:
                    viewer.render(options)
            else:
                self.viewer.render(options)

 
    def collect(self, criteria: Union[type, str], out: list[Component] = None) -> list[Component]:
        """
        Collect components by type, name.

        Parameters
        ----------
        criteria : type | str | Module
            Criteria to match component (type, name).

        Returns
        -------
        list[Component]
            List of matching components.
        """
        if out is None:
            out = []
        if isinstance(criteria, type):
            if type(self)==criteria or isinstance(self, criteria):
                out.append(self)
        elif isinstance(criteria, str):
            if criteria==seld.name:
                out.append(self)
        elif criteria==self:
            out.append(self)
        return out
   
 
    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Log state.

        Parameters
        ----------
        options : dict, optional
            Additional logging configuration.
        """
        pass
               
    def log_monitor(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Log monitor.

        Parameters
        ----------
        options : dict, optional
            Additional logging configuration.
        """
        if self.monitor:
            self.monitor.log(options)
            
class Module(Component):
    """
    Abstract computational module extending `Component`.

    Represents a single processing unit capable of propagating signals
    and performing step-based updates. Designed to be subclassed.

    Parameters
    ----------
    name : str, optional
        Name of the module.
    params : dict, optional
        Module configuration parameters.
    auto_sample : bool, optional
        Whether to automatically sample and trigger post-step callbacks.
    monitor : object or list, optional
        Monitor object(s) for data collection.
    viewer : object or list, optional
        Viewer object(s) for visualization.
    callback : callable or list, optional
        Function(s) invoked after each computation step.

    Methods
    -------
    step(s)
        Perform a forward computation step and optionally trigger sampling.
    propagate(s)
        Compute output for a given input. To be implemented in subclasses.
    """
    
    
    def __init__(
        self,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        auto_sample: bool = False,
        monitor: Optional[MonitorType] = None,
        viewer: Optional[ViewerType] = None,
        callback: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        super().__init__(name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer, callback=callback)

    def step(self, s: Any) -> Any:
        """
        Execute a single computation step.

        Parameters
        ----------
        s : any
            Input signal, state, or data for the module.

        Returns
        -------
        y : any
            Output signal resulting from propagation.
        """        
        y = self.propagate(s)
        if self.auto_sample:
            self.sample()
            self.post_step(s, y)
        return y

    def __call__(self, s: Any) -> Any:
        """Alias for `step(s)`."""        
        return self.step(s)

    def propagate(self, s: Any) -> Any:
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
        return None


class Fan(Module):
    """
    Module that fans out an input signal to multiple submodules.

    Calls `step()` on each referenced submodule with the same input
    and collects their outputs.

    Parameters
    ----------
    refs : list of Module
        Submodules to which the input signal is propagated.

    Returns
    -------
    list
        List of outputs from each submodule.
    """
    
    def __init__(self, refs: List[Module]) -> None:
        super().__init__()
        self.refs = refs

    def step(self, s: Any) -> List[Any]:
        """
        Propagate the input signal `s` to all referenced modules.

        Parameters
        ----------
        s : any
            Input signal shared across all submodules.

        Returns
        -------
        list
            List of outputs `y_i` from each submodule in `self.refs`.
        """
        yy = []
        for m in self.refs:
            y = m.step(s)
            yy.append(y)
        super().post_step(s)
        return yy


class Composite(Module):
    """
    Module composed of multiple submodules.

    Enables hierarchical composition, allowing grouped execution,
    logging, rendering, and updates across all contained components.

    Parameters
    ----------
    refs : list of Module
        List of submodules contained in this composite.
    name : str, optional
        Name of the composite module.
    callback : callable or list, optional
        Optional callback(s) triggered after each step.
    """
    
    def __init__(
        self,
        refs: Optional[List[Module]],
        name: Optional[str] = None,
        callback: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        super().__init__(name=name, callback=callback)
        self.refs = refs
        if refs is not None:
            for ref in refs:
                ref._parent = self

    def find(self, ref: Union[type, str, Module]) -> Optional[Module]:
        """
        Find a submodule by type, name, or reference.

        Parameters
        ----------
        ref : type | str | Module
            Target module type, name, or instance.

        Returns
        -------
        Module or None
            Matching module if found, otherwise None.
        """
        if self.refs is not None:
            for ref_ in self.refs:
                if isinstance(ref, type):
                    return ref_
                if isinstance(ref, str):
                    if ref==ref_.name:
                        return ref_
                if ref==ref_:
                    return ref_
        return None
 
    def collect(self, criteria: Union[type, str], out: list[Component] = None) -> list[Component]:
        """
        Collect submodules or components by type, name.

        Parameters
        ----------
        criteria : type | str | Module
            Criteria to match component (type, name).

        Returns
        -------
        list[Component]
            List of matching components.
        """
        if out is None:
            out = []
        if self.refs is not None:
            for ref_ in self.refs:
                ref_.collect(criteria, out)
        return out
    
    def dump(self) -> None:
        """
        Recursively call `dump()` on all submodules.
        """
        for m in self.refs:
            m.dump()

    def log(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Log internal state of all submodules.

        Parameters
        ----------
        options : dict, optional
            Additional logging configuration.
        """
        for ref in self.refs:
            if options is not None:
                _types = options.get('types', None)
                if _types is not None:
                    b = False
                    for _type in _types:
                        if isinstance(ref, _type):
                            b = True
                            break
                    if not b:
                        continue 
                _names = options.get('names', None)
                if _names is not None:
                    if ref.name is None or ref.name not in _names:
                        continue 
            ref.log(options=options)

    def log_monitor(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Log monitor of all submodules.

        Parameters
        ----------
        options : dict, optional
            Additional logging configuration.
        """
        super().log_monitor(options)
        for ref in self.refs:
            ref.log_monitor(options=options)

    def update(self, s: Any, y: Any) -> None:
        """
        Propagate an update signal to all submodules.

        Parameters
        ----------
        s : any
            Input signal or state.
        y : any
            Output or feedback signal.
        """
        super().update(s, y)
        for ref in self.refs:
            ref.update(s, y)
    
    def sample(self) -> None:
        """
        Trigger sampling on the composite and all submodules.
        """
        super().sample()
        for ref in self.refs:
            ref.sample()
    
    def render(self, options: Any) -> None:
        """
        Render visualization for all contained submodules.

        Parameters
        ----------
        options : dict or any
            Rendering configuration passed to each submodule.
        """
        super().render(options)
        for ref in self.refs:
            ref.render(options)
            
class Chain(Composite):
    """
    Sequential composite module chaining multiple submodules.

    Executes submodules in order, feeding the output of each into the next.
    Optionally computes input/output shape compatibility for the chain.

    Parameters
    ----------
    refs : list of Module
        Ordered list of submodules forming the processing chain.
    name : str, optional
        Name of the chain module.
    callback : callable or list, optional
        Optional callback(s) triggered after each chain step.
    """
    
    def __init__(
        self,
        refs: List[Module],
        name: Optional[str] = None,
        callback: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        super().__init__(refs=refs, name=name, callback=callback)
        self._shape()
     
    def _shape(self) -> Optional[Tuple[int, int]]:         
        """
        Infer overall input/output shape of the module chain.

        Returns
        -------
        tuple or None
            Shape as (output_dim, input_dim), or None if unknown.
        """
        self.shape = None
        if isinstance(self.refs, list) and len(self.refs)>0:
            if self.refs[0].shape is not None:
                if len(self.refs)>1:
                    n = self.refs[0].shape[-1]
                    m = self.refs[-1].shape[0]
                    self.shape = (m, n)  
                else:
                    self.shape = self.refs[0].shape
        return self.shape
            
    def step(self, s: Any) -> Any:
        """
        Sequentially propagate the signal through all submodules.

        Parameters
        ----------
        s : any
            Input signal for the first submodule.

        Returns
        -------
        y : any
            Final output after passing through the entire chain.
        """
        y = s
        for m in self.refs:
            y = m.step(y)
        super().post_step(s, y)
        return y

