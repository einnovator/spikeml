from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from spikeml.core.monitor import Monitor
from spikeml.core.viewer import MonitorViewer
from spikeml.core.snn import Connector


from spikeml.core.snn_stats import connector_stats

from spikeml.plot.plot_stats import plot_stats_matrix


def plot_connector_stats(results):
    conns = results.get_connectors()
    for conns_ in  conns:
        m, sd = connector_stats(conns_)
        plot_stats_matrix(m, sd, title=conns_[0].name)

def plot_stats(results):
    plot_connector_stats(results)
