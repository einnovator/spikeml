import math
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Optional, Union, List, Tuple


from spikeml.core.snn_monitor import ErrorMonitor
from spikeml.core.snn_viewer import ErrorMonitorViewer
from spikeml.core.feedback import compute_error, compute_sg
from spikeml.core.spikes import spike

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
    plot: bool = True,
    callback: Optional[Callable[[Context], bool]] = None,
    log_step: int = 1,
    silent: bool = False,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run a single neural network simulation over time.

    Args:
        nn: Neural network instance with methods `__call__`, `sample`, `log`, and `render`.
        ss (np.ndarray): Input stimulus array (shape: [T, N] or [N]).
        T (int, optional): Number of time steps. Defaults to inferred from `ss` if not given.
        params: Simulation or model parameters (must define attributes `vmin`, `vmax`, `e_err`, `g`).
        feedback (bool, optional): Whether to enable error feedback loop. Defaults to True.
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

    if not silent:
        if feedback:
            err_monitor.log()
        nn.log_monitor(options)

    if plot:
        nn.render(options)
        err_viewer.render()

    result = { 'nn': nn, 'err_monitor': err_monitor, 'err_viewer': err_viewer, 't': t }
    return result

def nrun(
    nn_creator: Callable[[int, int], Any],
    ss: np.ndarray,
    runs: int = 1,
    T: Optional[int] = None,
    params: Optional[Any] = None,
    feedback: bool = True,
    callback: Optional[Callable[[Context], bool]] = None,
    plot: bool = False,
    log_runs: int = 1,
    log_step: int = -1,
    options: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """Run multiple neural network simulations sequentially.

    Args:
        nn_creator (Callable[[int, int], Any]): Function to create a neural network instance for each run.
            Receives `(run_index, total_runs)` as arguments.
        ss (np.ndarray): Input stimulus array (shape: [T, N] or [N]).
        runs (int, optional): Number of runs to perform. Defaults to 1.
        T (int, optional): Number of timesteps per run. Defaults to inferred from `ss`.
        params: Simulation parameters passed to each run.
        feedback (bool, optional): Enable feedback loop. Defaults to True.
        callback (Callable[[Context], bool], optional): Callback executed at each timestep.
        plot (bool, optional): Whether to plot results after each run. Defaults to False.
        log_runs (int, optional): Frequency of per-run logging. Defaults to 1.
        log_step (int, optional): Logging interval during simulation steps. Defaults to -1 (disabled).
        options (dict, optional): Extra run options.
        debug (bool, optional): Enable detailed logging output. Defaults to False.

    Returns:
        list[dict]: List of result dictionaries (same structure as returned by `run()`).
    """
    results = []
    for n in range(0,runs):
        debug_ = debug and (log_runs>0 and n%log_runs==0)
        if debug_:
            print(f'run: {n+1}/{runs}')
        nn = nn_creator(n, runs)
        result = run(nn, ss, T=T, params=params, DY = 0, feedback=True, callback=callback, plot=plot, log_step=log_step, options=options, debug=debug)
        if debug_:
            print(f'  -> {result}')
        results.append(result) 
    return results  

