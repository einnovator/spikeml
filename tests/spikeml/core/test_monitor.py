from spikeml.core.base import Component, Module, Fan, Composite, Chain
from spikeml.core.monitor import Monitor


def test_monitor():
    monitor = Monitor()
    print(monitor)
    
if __name__ == '__main__':  
    test_monitor()