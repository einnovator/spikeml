
from spikeml.core.params import Params, NNParams, ConnectorParams, SpikeParams, SSensorParams, SNNParams, SSNNParams

def test_params():
    print(SSNNParams(g=4, pmax=.5))
    params = SSNNParams(g=4)
    params.g=5
    print(params)
    print(params.fmt())
    fields = params.__class__.model_fields
    for key, field in fields.items():
        print(key, field, field.annotation, field.annotation==float, field.default, field.metadata)
        

if __name__ == '__main__':  
    test_params()