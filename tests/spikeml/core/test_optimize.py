   
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
from pydantic import BaseModel

from spikeml.core.params import SSNNParams
from spikeml.core.feedback import compute_error, compute_sg

from spikeml.core.spikes import pspike, spike


def test_make_params_spec():        
    params = SSNNParams()
    print(params)
    set_all_attrs(params, None, exclude=['g', 't_p', 't_d', 'pmax'])
    print(params)
    spec = make_params_spec(params)
    print(spec)     
    
def test_params_search():
    def f(params):
        params = setattrs(SSNNParams(), params)
        return params.g
    
    params = set_all_attrs(SSNNParams(), None, exclude=['g', 't_p', 't_d', 'pmax'])
    spec = make_params_spec(params)
    params, err = params_search(spec, f, debug=True)
    

if __name__ == '__main__':  
    test_make_params_spec()
    test_params_search()