
from spikeml.core.snn import bias_update

def test_bias_update():
    params = SSNNParams(t_b=10)
    print(params)
    def _bias_update(b, y, n=10):
        for i in range(0,n):
            b = bias_update(b, y, params, debug=True)
    
    _bias_update(np.array([1]), np.array([0]))
    print('-'*10)
    _bias_update(np.array([0]), np.array([1]))

if __name__ == '__main__':  
    test_bias_update()