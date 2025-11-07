import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_stats_matrix(m, sd, title=None, _type=None, callback=None, ax=None):

    xx = [ f'[{i},{j}]' for i in range(m.shape[0]) for j in range(m.shape[0])]
            
    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 5*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]
            
    y = m.flatten()
    yerr = sd.flatten() if sd is not None else None
    #print('m:', m, 'sd:', sd)
    #print('xx:', xx, 'y:', y, 'yerr:', yerr)

    plt.errorbar(xx, y, yerr=yerr, capsize=3, fmt="r--o", ecolor = "black")
        
    if callback is not None:
        callback(ax)
    
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)

    #ax.legend(fontsize=6)
    #ax.tick_params(axis='both', which='major', labelsize=6)
    #ax.tick_params(axis='both', which='minor', labelsize=6)
    #if xlabel is not None:
    #    ax.set_xlabel(xlabel, fontsize=6)
    #if ylabel is not None:
    #    ax.set_xlabel(ylabel, fontsize=6)

    if fig is not None:
        #plt.tight_layout()
        plt.show()
        


def plot_stats_matrix__(results, title=None, _type=None, callback=None, ax=None):

    def _get_M(result):
        if isinstance(result, np.ndarray):
            M = result
        else:
            M = result.M
        return M        

    m = None
    for n, result in enumerate(results):
        M = _get_M(result)
        m = M.copy() if m is None else m+M
    m /= len(results) 
    sd2 = np.zeros(m.shape)
    for n, result in enumerate(results):
        M =  _get_M(result)
        sd2 += (m-M)**2
    sd2 /= len(results)
    sd = np.sqrt(sd2)

    xx = [ f'[{i},{j}]' for i in range(m.shape[0]) for j in range(m.shape[0])]
            
    fig = None
    if ax is None:
        nrows, ncols = 1, 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 5*nrows))
        ax = axs if nrows==1 and ncols==1 else axs[i] if nrows==1 else axs[0][0]
            
    y = m.flatten()
    yerr = sd.flatten()
    #print('m:', m, 'sd:', sd)
    #print('xx:', xx, 'y:', y, 'yerr:', yerr)

    plt.errorbar(xx, y, yerr=yerr, capsize=3, fmt="r--o", ecolor = "black")
        
    if callback is not None:
        callback(ax)
    
    if title is not None:
        tstyle = { 'fontsize': 8 }
        ax.set_title(title, **tstyle)

    #ax.legend(fontsize=6)
    #ax.tick_params(axis='both', which='major', labelsize=6)
    #ax.tick_params(axis='both', which='minor', labelsize=6)
    #if xlabel is not None:
    #    ax.set_xlabel(xlabel, fontsize=6)
    #if ylabel is not None:
    #    ax.set_xlabel(ylabel, fontsize=6)

    if fig is not None:
        #plt.tight_layout()
        plt.show()
        

