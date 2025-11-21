import math
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Optional, Union, List, Tuple


from spikeml.core.base import Component
from spikeml.core.snn_monitor import ErrorMonitor
from spikeml.core.snn_viewer import ErrorMonitorViewer
from spikeml.core.feedback import compute_error, compute_sg
from spikeml.core.spikes import spike
from spikeml.core.snn import Connector, SimpleLayer

class Context(BaseModel):
    """Execution context for neural network simulations.

    Attributes:
        t (int): Current simulation timestep (>0).
    """
    t: int = Field(default=0, gt=0)

    def __str__(self) -> str:
        """Return a string representation of the context."""
        return f'{type(self).__name__}({vars(self)})'     
    
def run(
    nn: Any,
    ss: np.ndarray,
    T: Optional[int] = None,
    params: Optional[Any] = None,
    feedback: bool = True,
    report: bool = True,
    plot: bool = True,
    log_step: int = 1,
    callback: Optional[Callable[[Context], bool]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run a single neural network simulation over time.

    Args:
        nn: Neural network instance with methods `__call__`, `sample`, `log`, and `render`.
        ss (np.ndarray): Input stimulus array (shape: [T, N] or [N]).
        T (int, optional): Number of time steps. Defaults to inferred from `ss` if not given.
        params: Simulation or model parameters (must define attributes `vmin`, `vmax`, `e_err`, `g`).
        feedback (bool, optional): Whether to enable error feedback loop. Defaults to True.
        report (bool, optional): Whether to output run report. Defaults to True.
        plot (bool, optional): Whether to render plots after the run. Defaults to True.
        callback (Callable[[Context], bool], optional): Function called at each timestep.
            Return False to stop simulation early.
        log_step (int, optional): Frequency of log output (in steps). Defaults to 1.
        options (dict, optional): Extra configuration flags for logging and rendering.

    Returns:
        dict: A dictionary containing simulation results:
            - `'nn'`: Final neural network state
            - `'err_monitor'`: Error monitor instance (if feedback enabled)
            - `'err_viewer'`: Error viewer instance (if feedback enabled)
            - `'t'`: Final timestep index
    """
    if len(ss.shape)==2:
        s0 = ss[0]
        if T is None:
            T = ss.shape[0]
    else:
        s0 = ss
    if T is None:
        T = 10
    sx = s0

    err_monitor,err_viewer = None,None
    if feedback:
        err_monitor = ErrorMonitor(ref=nn)
        err_viewer = ErrorMonitorViewer(err_monitor)
        err_monitor.viewer = err_viewer
        
    done = False
    s = sx
    zs = np.zeros(s.shape)
    
    context = Context()
    for t in range(0, T):
        debug_ = log_step>=0 and (t==0 or t==T-1 or (log_step>0 and t%log_step==0))
        
        if debug_:
            if options is None or options.get('log.time', True):
                print(f't={t}')
            
        context.t = t
        s_ = (s, sx)
        y,zy = nn(s_)
        nn.sample()

        sg = 1
        if feedback:
            #err = compute_error(sx, y)
            #err = xcompute_error(sx, y, R=R, method='sum+clip')
            err = compute_error(sx, zy)
            sg = compute_sg(err, params)
            err_monitor.sample(sx, err, sg)

        if callback is not None:
            if callback(context)==False:
                done = True

        sx = ss[t % ss.shape[0]] if len(ss.shape)==2 else s0
        s = sx
        sy = None
        if feedback:
            sy = params.g*zy #y
            s = sx + sy
            s *= sg
            s = np.clip(s, params.vmin, params.vmax)

        if debug_:
            nn.log(options)
            if feedback:
                if options is None or options.get('log.err', True):
                    print(f'sx={sx}; sy={sy}; sg={sg:.2f}; s={s}; err={err:.3f}')
            print('-'*10)

        if done:
            break

    if report:
        if feedback:
            err_monitor.log()
        nn.log_monitor(options)

    if plot:
        nn.render(options)
        err_viewer.render()

    result = { 'nn': nn, 'err_monitor': err_monitor, 'err_viewer': err_viewer, 'context': context }
    return result

def nrun(
    nn_creator: Callable[[int, int], Any],
    ss: np.ndarray,
    runs: int = 1,
    T: Optional[int] = None,
    params: Optional[Any] = None,
    feedback: bool = True,
    callback: Optional[Callable[[Context], bool]] = None,
    report_run: bool = False,
    plot_run: bool = False,
    report: bool = False,
    plot: bool = False,
    log_runs: int = 1,
    log_step: int = -1,
    options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Execute multiple neural network simulations sequentially.

    This function repeatedly creates and runs neural network instances using a
    user-supplied constructor. Each simulation can share the same input stimulus
    but operate with independent network initializations or parameters.

    Parameters
    ----------
    nn_creator : callable
        Function that creates a neural network instance for a given run.
        Called as `nn_creator(run_index, total_runs)` for each run.
    ss : ndarray
        Input stimulus array, of shape `(T, N)` or `(N,)`, representing the
        time series or feature inputs to the network.
    runs : int, optional
        Number of independent runs to execute. Default is 1.
    T : int, optional
        Number of time steps per run. If `None`, inferred from `ss`.
    params : Any, optional
        Simulation parameters passed to each run.
    feedback : bool, optional
        Whether to enable network feedback during simulation. Default is True.
    callback : callable, optional
        Optional function called at each simulation step with the current
        `Context` object. If it returns `False`, the run may terminate early.
    report_run : bool, optional
        Whether to display a textual report after each run. Default is False.
    plot_run : bool, optional
        Whether to plot intermediate results after each run. Default is False.
    report : bool, optional
        Whether to output a combined summary report after all runs. Default is False.
    plot : bool, optional
        Whether to generate final plots for aggregated results. Default is False.
    log_runs : int, optional
        Frequency (in runs) at which progress and debug logs are printed.
        Default is 1 (log every run).
    log_step : int, optional
        Interval of step-level logging within each run. `-1` disables step logging.
        Default is -1.
    options : dict, optional
        Additional keyword arguments passed to each call of `run()`.

    Returns
    -------
    results : list of dict
        List containing the result dictionary from each run, matching the
        structure returned by the underlying `run()` function.

    Notes
    -----
    - Each run creates a fresh network instance using `nn_creator`.
    - Useful for performing Monte Carlo simulations or parameter sweeps.
    - Logging can be controlled via `log_runs` and `log_step`.

    Examples
    --------
    >>> def create_net(i, total):
    ...     return make_ssnn_chain(name=f'nn{n}', size=size, params=params)
    >>> ss = signal_pulse(2, T=100, L=3, s=[0,1], value=.5)
    >>> params = SSNNParams()
    >>> results = nrun(nn_creator= create_net, runs=3, ss=ss, params=params)
    >>> len(results)
    """
    results = []
    for n in range(0,runs):
        debug_ = (log_runs>0 and n%log_runs==0)
        if debug_:
            print(f'run: {n+1}/{runs}')
        nn = nn_creator(n, runs)
        result = run(nn, ss, T=T, params=params, feedback=feedback, callback=callback, report=report_run, plot=plot_run, log_step=log_step, options=options)
        if debug_:
            print(f'  -> {result}')
        results.append(result) 
    if report:
        pass
    if plot:
        pass
    
    return Results(results)



class Results():
    """Results from Multiple Runs.
    """
    results = []
    def __init__(self,
                 results: list = None,
    ) -> None:
        #super().__init__()
        self.results = results

    def iterate(self, callback, *args) -> None:
        if self.results is not None:
            for n, result in enumerate(self.results):
                callback(result, *args)
        
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
        if self.results is not None:
            for n, result in enumerate(self.results):
                obj = result['nn']
                out_ = obj.collect(criteria)
                out.append(out_)
        return out

    def __getitem__(self, index):
        return self.results[index]
    
    def len(self):
        return len(self.results) if self.results is not None else 0

    def __len__(self):
        return self.len()

    def log_monitor(self):
        for n, result in enumerate(self.results):
            result['err_monitor'].log()

    def plot_monitor(self, options=None):
        for n, result in enumerate(self.results):
            result['nn'].viewer.render(options=options)

    def plot_err_monitor(self, options=None):
        for n, result in enumerate(self.results):
            result['err_viewer'].render(options=options)
            
    def __repr__(self):
        return f"{type(self).__name__}({self.results!r})"

    def get_components(self, _type, as_map=False):
        objs = self.collect(_type)
        objsT = list(map(list, zip(*objs)))
        if as_map:
            out = {}
            for i,layer_runs in enumerate(objsT):
                obj = layer_runs[0]
                name = obj.name
                if name is None:
                    name =  f"{type(obj).__name__}.{i}"
                out[name] = layer_runs
            return out
        return objsT

    def get_connectors(self, as_map=False):
        return self.get_components(Connector, as_map=as_map)
    
    def get_layers(self, as_map=False):
        return self.get_components(SimpleLayer, as_map=as_map)

    def get_connector_tensors(self, as_map=False):
        conns = self.get_connectors(as_map)
        if as_map:
            MM = { name : np.array([c.M for c in layer_runs]) for name, layer_runs in conns.items() }
        else:
            MM = [ np.array([c.M for c in layer_runs]) for layer_runs in conns]
        return MM
