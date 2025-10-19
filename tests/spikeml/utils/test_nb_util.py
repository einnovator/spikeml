
import numpy as np

from spikeml.utils.nb_util import xdisplay, Markup

def test_xdisplay():
    A = np.array([[1/3, 2, 3], [3, 4,3],[1/3, 2, 3], [3, 4,3]])
    B = np.array([[5, 6, 5], [7, 8, 5],[5, 6, 5], [7, 8, 5]])

    xdisplay(Markup('t=0', np.zeros(5), np.ones(5)), Markup('A', A), Markup('B', B),  Markup('AB', (A,B)), np.zeros(5), [1,2,3], (1,2,3), "abc", 3, 1/3)

