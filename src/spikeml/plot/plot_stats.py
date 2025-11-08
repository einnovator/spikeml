import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import cm

def plot_stats_matrix(m, sd, data=None, title=None, _type=None, callback=None, ax=None):

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

    ax.errorbar(xx, y, yerr=yerr, capsize=3, fmt="r--o", ecolor = "black")
    if data is not None:
        n = data.shape[1]*data.shape[2]
        colors = cm.get_cmap('tab20', data.shape[0])
        for i in range(data.shape[0]):
            ax.scatter(xx, data[i].flatten(), color=colors(i), marker='s', s=30, alpha=0.9)
          
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
        

