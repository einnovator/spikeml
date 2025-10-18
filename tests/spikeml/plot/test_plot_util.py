
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt

from spikeml.plot.plot_util import plot_hist, plot_data, plot_lidata, plot_input, plot_xt, plot_mt, plot_spikes, imshow_matrix, imshow_nmatrix, 


def test_plot_data():
    data0 = [t**1.5 for t in range(0,10)]
    data1 = [t**2 for t in range(0,10)]
    data = {'data0': data0, 'data1': data1 }
    ss = signal_pulse(2, T=3, L=1, s=[0,1,-1], value=1)
    plot_data(data, title='yy', callback=lambda ax: plot_input(ss,ax=ax))
    data = [np.array([1*t,2*t])**1.5 for t in range(0,10)]
    plot_data(data, title='yy2', label='yy')#['y1','y2']
    plot_data(np.array(data), title='yy2.2', label='yy')#['y1','y2']

def test_plot_lidata():
    t = np.linspace(0, 100, num=100)
    x = np.abs(np.sin(t)) + np.random.normal(loc=0, scale=.2, size=t.shape[0])
    k_x = .8
    plot_lidata(x, k_x)
            
def test_imshow_matrix():
    m = np.random.normal(loc=0, scale=1, size=(5,5))
    imshow_matrix(m, title='M')

def test_imshow_nmatrix():
    data = [ np.random.normal(loc=0, scale=1, size=(5,5)) for i in range(0,16) ]
    imshow_nmatrix(data, title='M', tstep=3)
    imshow_nmatrix(data, title='M', tk=10)
    imshow_nmatrix(data, title='M', tk=10, ncols=10)

if __name__ == '__main__':  
    test_plot_data()
    test_plot_lidata()
    test_imshow_matrix()
    test_imshow_nmatrix()