from spikelogiks.core import Neuron

def test_spike():
    n = Neuron(threshold=1.0)
    assert not n.step(0.5)
    assert n.step(0.6)  # should spike
