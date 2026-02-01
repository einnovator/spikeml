import numpy as np


from spikeml.utils.img_util import show_img, show_imgs, cv_bgr2rgb, mshow, show_kernels, show_kernels_

from spikeml.utils.nb_util import xdisplay, Markup

def show_kmaps(xx, ym, labels, M=None, klabels=False, batch=True, debug=False):
    if debug:
        xdisplay(*[ Markup(f'{klabels[j]}: M[{j}]', M[j]) for j in range(M.shape[0])])
    if M is not None:
        kimgs = [None] + list(M)
        if klabels==True:
            klabels = [f'{i}' for i in range(0, M.shape[0])]   
        elif klabels==False:
            klabels = None        
        if isinstance(klabels, list):
            klabels = [None]+klabels
            show_kernels(kimgs, pad=0.1, figsize=((1+M.shape[0])*1, 1), titles=klabels)
        nk = M.shape[0]
    else:
        nk = xx.shape[1 if batch else 0]
    if batch:
        for i in range(ym.shape[0]):
            #show_tiles(xt[i], titles=labels_[i], batch=False, figsize=(1,1), pad=0.1, normalize=False)
            if debug:
                xdisplay(*([Markup(f'{labels[i]} xx[{i}]', xx[i])]+[ Markup(f'ym[{i},{j}]', ym[i,j]) for j in range(ym.shape[1])]))
            imgs = [xx[i]] + list(ym[i])
            show_imgs(imgs, ncols=1+min(nk,20), figsize=(len(imgs)*1,1), pad=.1,
                    titles=[labels[i]] if labels is not None else None, title_size=6, normalize=False)
            if debug:
                print('--'*10)
    else:
        if debug:
            xdisplay(*([Markup(f'{labels} xx', xx)] + [ Markup(f'ym[{j}]', ym[j]) for j in range(ym.shape[0])]))
        imgs = [xx] + list(ym)
        show_imgs(imgs, ncols=1+min(nk,20), figsize=(len(imgs)*1,1), pad=.1,
                titles=[labels], title_size=6, normalize=False)



def show_tiles(xt, figsize=None, ncols=None, nrows=None, batch=True, tilesize=.5, pad=0.05, titles=None, normalize=True, subplots=False):

    def _show_tiles(xi, suptitle=None, parent=None, fig=None):
        xi_ = xi.reshape(-1, *xi.shape[2:])
        nonlocal figsize
        if figsize is None and parent is None:
            figsize = (tilesize*xi.shape[1], tilesize*xi.shape[0])
        show_imgs(xi_, suptitle=suptitle, titles=None, nrows=xi.shape[0], ncols=xi.shape[1], title_size=6, 
                  figsize=figsize, pad=pad, normalize=normalize, parent=parent, fig=fig)

    if batch:
        n = xt.shape[0]
        if subplots:
            if ncols==None:
                if nrows==None:
                    ncols = min(20, n)
                else:
                    ncols=int(n/nrows) + (n % nrows > 0)
            if nrows==None:
                nrows=n//ncols + int(n% ncols!=0)
            fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*tilesize*xt.shape[2], nrows*tilesize*.85*xt.shape[1]))
            axs = axs.flatten() if (nrows>1 or ncols>1) else [axs]
            pad /= 2
        else:
            fig = None
        for i in range(0, xt.shape[0]):
            xi = xt[i]
            suptitle = None
            if subplots:
                parent = axs[i].get_subplotspec()
                axs[i].remove()
            else: 
                parent = None
            if titles is not None :
                suptitle = titles[i]
            _show_tiles(xi, suptitle=suptitle, parent=parent, fig=fig)
        
        if fig is not None:
            #plt.tight_layout()
            plt.show()
    else:
        _show_tiles(xt, suptitle=titles)
        
