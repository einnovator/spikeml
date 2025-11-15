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
        fig, axs = plt.subplots(nrows, ncols, figsize=(1*len(xx), 3*nrows))
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
        

def plot_dist_matrix(data=None, title=None, callback=None, ax=None):
    if ax is None:
        ax = plt
    fig, axs = ax.subplots(data.shape[1], data.shape[2], figsize=(1*data.shape[2], 1*data.shape[1]))
                
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ax_ = axs if data.shape[1]==1 and data.shape[2]==1 else axs[i] if data.shape[1]==1 else axs[i][j]
            data_ = data[:,i,j]
            mean = np.mean(data_)
            sd  = np.std(data_)  
            bins = min(data.shape[0],10)
            counts, bin_edges = np.histogram(data_, bins=bins)
            ax_.hist(data_, bins=bins, density=True, alpha=0.5, color="blue", edgecolor=None)
            idx = np.argmax(counts)
            bin_left = bin_edges[idx]
            bin_right = bin_edges[idx+1]
            md = 0.5 * (bin_left + bin_right)
            ax_.axvline(mean, color='red', linewidth=.5, label=f"{mean:.2f}")
            ax_.axvline(mean - sd, color='green', linestyle='--', linewidth=.5)
            ax_.axvline(mean + sd, color='green', linestyle='--', linewidth=.5)
            #ax_.text(mean, ax_.get_ylim()[1]*0.9, f"{mean:.2f}", color='red', ha='center', va='top', fontsize=5)
            #ax_.text(mean - sd, ax_.get_ylim()[1]*0.75, f"{mean - sd:.2f}", color='green', ha='center', va='top', fontsize=5)
            #ax_.text(mean + sd, ax_.get_ylim()[1]*0.75, f"{mean + sd:.2f}", color='green', ha='center', va='top', fontsize=5)
            ax_.set_xticks([])
            ax_.set_yticks([])
            #ax_.set_axis_off()
            ax_.spines['bottom'].set_alpha(0.5)
            ax_.spines['left'].set_alpha(0.5)
            ax_.spines['top'].set_alpha(0.5)
            ax_.spines['right'].set_alpha(0.5)
            s = rf"$\mu={mean:.2f}$"
            b = idx > bins/2
            x = 0.02 if b else (1-0.02)
            ha = 'left' if b else 'right'
            ax_.text(x, .98, s, transform=ax_.transAxes, ha=ha, va='top', fontsize=5, color='red')
            s = rf"$\sigma={sd:.2f}$"
            ax_.text(x, .88, s, transform=ax_.transAxes, ha=ha, va='top', fontsize=5, color='green')
        
    if title is not None:
        tstyle = { 'fontsize': 8 }
        plt.suptitle(title, **tstyle)

    if fig is not None:
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
