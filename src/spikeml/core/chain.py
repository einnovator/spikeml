
import numpy as np
from typing import Optional, Tuple, Union, Any, Dict

from spikeml.utils.vector import _sum, upsample
from spikeml.core.base import Component, Module, Fan, Composite, Chain
from spikeml.core.params import SSensorParams, Params, NNParams, ConnectorParams, SpikeParams, SNNParams, SSNNParams
from spikeml.core.snn import SSensor, SNN, SSNN, LIConnector

from spikeml.core.matrix import matrix_init, matrix_init2

def make_chain(constructor, input_contructor=None, output_contructor=None, size=None, k=1, name='nn', input_params=None, params=None, auto_sample=True, monitor=True, viewer=True):
    nns = []
    if params is None:
        params = SSNNParams()
    _size = None     
    if input_contructor is not None:
        if input_params is None:
            input_params = SSensorParams.model_validate(params.model_dump())        
        name_ = f'{name}.s' if input_params.name is None else input_params.name        
        size_ = size[0] if isinstance(size, list) else size
        sensor = input_contructor(name_, size_, input_params)
        nns.append(sensor)
        _size = size_
        i0 = 1
        if isinstance(size, list):
            k = len(size)-1
    else:
        i0 = 0
        if isinstance(size, list):
            k = len(size)   
    for k_ in range(0, k):
        iparams = params[k_] if isinstance(params, list) else params
        name_ = f'{name}.{k_+i0}' if iparams.name is None else iparams.name
        size_ = size[k_+i0] if isinstance(size, list) else size
        if not isinstance(size_, tuple):
            if _size is None:
                _size = size_
            size_ = (size_,_size)
        nnk = constructor(name_, size_, iparams)
        _size = size_[0] if isinstance(size_, tuple) else size_
        nns.append(nnk)
    nn = Chain(nns, name=name) 
    return nn


def chain_validate(nn):
    _size = None
    print(f'{nn.name} {nn} :')    
    for ref in nn.refs:
        ok = _size is None or ref.shape[-1]==_size
        print(f'  {ref.name} {ref} :', ref.shape, 'OK' if ok else 'ERR', 'OK' if nn.find(type(ref))!=None else 'NOT_FOUND')
        _size = ref.shape[0]

def make_ssnn_chain(k=1, size=None, name='nn', params=None, sensor_params=None, auto_sample=True, monitor=True, viewer=True):
    def _layer(name, size, params):
        M = LIConnector(size=size, name=name, params=params, monitor=monitor, viewer=viewer)
        return SSNN(name=name, M=M, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

    def _sensor(name, size, params):
        return SSensor(name=name, n=size, params=sensor_params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

    return make_chain(_layer, _sensor, k=k, size=size, name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

def make_snn_chain(k=1, size=None, name='nn', params=None, auto_sample=True, monitor=True, viewer=True):
    def _layer(name, size, params):
        M = LIConnector(size=size, name=name, params=params, monitor=monitor, viewer=viewer)
        return SNN(M=M, name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

    def _sensor(name, size, params):
        return SSensor(name=name, n=size, params=sensor_params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

    return make_chain(_layer, k=k, size=size, name=name, params=params, auto_sample=auto_sample, monitor=monitor, viewer=viewer)

