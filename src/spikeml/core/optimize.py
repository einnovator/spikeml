import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
from pydantic import BaseModel

from spikeml.core.params import Params, NNParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

def make_params_spec(
    params: BaseModel,
    keys: Optional[Union[List[str], Dict[str, Any]]] = None,
    skip: bool = True
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Generate a specification dictionary of parameter search ranges from a Pydantic model.

    This function inspects a Pydantic model (e.g., `SSNNParams`) and extracts the default,
    minimum, and maximum values from field annotations and metadata (e.g., `Gt`, `Ge`, `Lt`, `Le`).
    It is primarily used to define parameter ranges for optimization or grid search.

    Args:
        params (BaseModel): The Pydantic model instance defining parameters.
        keys (list[str] | dict[str, Any], optional): Subset of parameter names to include.
        skip (bool, optional): Whether to skip parameters with `None` values. Defaults to True.

    Returns:
        dict[str, tuple[float, tuple[float, float]]]: A mapping of
            `{ parameter_name: (initial_value, (min_value, max_value)) }`.
    """
    from annotated_types import Gt, Lt, Ge, Le, Interval, MinLen, MaxLen, Len, MultipleOf
    
    spec = {}
    fields = params.__class__.model_fields    
    for name, field in fields.items():
        if keys is not None:
            if isinstance(keys, list):
                if name not in keys:
                    continue
            elif isinstance(keys, dict):
                if name not in keys.keys():
                    continue
        _type = field.annotation
        value = getattr(params, name)
        if value is None:
            if skip:
                continue
            value = field.default
            if value is None:
                continue
        #print(name, field)
        min_value = None
        max_value = None
        if field.metadata is not None:
            for meta in field.metadata:
                if isinstance(meta, Gt):
                    min_value = meta.gt
                elif isinstance(meta, Ge):
                    min_value = meta.ge
                elif isinstance(meta, Lt):
                    max_value = meta.lt
                elif isinstance(meta, Le):
                    max_value = meta.le
                #print(meta, type(meta)) #Annotated
        radius = None
        if radius is None:
            if min_value is not None:
                radius = value - min_value
            elif max_value is not None:
                radius = value + max_value
        if radius is not None:
            if min_value is None:
                min_value = value-radius
            if max_value is None:
                max_value = value+radius
        if min_value is None:
            continue
        if max_value is None:
            continue
        spec[name] = (value, (min_value, max_value))
    return spec

def vec2dic(x: Union[np.ndarray, List[float]], keys: List[str]) -> Dict[str, float]:
    """Convert a numeric vector to a dictionary using the given parameter keys.

    Args:
        x (np.ndarray | list[float]): Numeric vector of parameter values.
        keys (list[str]): Corresponding parameter names.

    Returns:
        dict[str, float]: Mapping of parameter names to their values.
    """
    return {key: float(x[i]) for i, key in enumerate(keys)}

def setattrs(model: Any, key_value: Dict[str, Any]) -> Any:
    """Set multiple attributes on a model or object.

    Args:
        model (Any): Object on which to set attributes.
        key_value (dict[str, Any]): Dictionary of attribute names and values.

    Returns:
        Any: The same model instance, modified in place.
    """
    for key, value in key_value.items():
        setattr(model, key, value)
    return model

def params_search(
    spec: Dict[str, Tuple[float, Tuple[float, float]]],
    callback: Callable[[Dict[str, float]], float],
    debug: bool = False
) -> Tuple[Dict[str, float], float]:
    """Run a parameter optimization using `scipy.optimize.minimize`.

    Args:
        spec (dict[str, tuple[float, tuple[float, float]]]): Parameter specification dict
            (as produced by `make_params_spec`).
        callback (Callable[[dict[str, float]], float]): Function that computes an error
            (to minimize) given a dict of parameter values.
        debug (bool, optional): If True, prints progress and intermediate values.

    Returns:
        tuple[dict[str, float], float]: Optimized parameters and final error value.
    """
    def f(x):
        params = vec2dic(x,spec.keys())        
        err = callback(params)
        if debug:
            print('>>', params, '->', err)
        return err
    
    x0 = np.array([spec[key][0] for key in spec.keys()])
    bounds = [spec[key][1] for key in spec.keys()]
    result = minimize(f, x0, bounds=bounds)
    params = vec2dic(result.x, spec.keys())
    err = result.fun
    if debug:
        print('>>', params, '->', err)
    return params, err


def set_all_attrs(
    model: Any,
    value: Any,
    keys: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Any:
    """Set the same value on all attributes of a model or object.

    Args:
        model (Any): Object or model instance.
        value (Any): Value to assign to each attribute.
        keys (list[str], optional): Specific attributes to set. Defaults to all attributes.
        exclude (list[str], optional): Attributes to skip. Defaults to None.

    Returns:
        Any: The same model instance with modified attributes.
    """
    if keys is None:
        keys = vars(model).keys()
    for key in keys:
        if exclude is not None:
            if key in exclude:
                continue
        setattr(model, key, value)
    return model

 