from IPython.display import display, HTML

import numpy as np

class Markup(object):
    def __init__(self, *args):
        self.args = args
        
def html(*args, **kwargs):
    decimals = 3
    font_size=11
    style = f'margin:0px;padding:0px;font-size:{font_size}px'
    sep_pad = 5
    tuple_pad = 0
    css_td = "padding:1px;padding-right:2px"
    css_lsep = "border-right:thin solid #000; padding:0px;padding-left:2px"
    css_rsep = "border-left:thin solid #000; padding:0px;padding-right:2px"
    css_ldsep = "border-right:2px solid #000; padding:0px;padding-left:2px"
    css_lddsep = "border-right:4px solid #000; padding:0px;padding-left:2px"
    css_ldddsep = "border-right:6px solid #000; padding:0px;padding-left:2px"

    def np2html(a):
        """" Convert arrays (vectors, matrices, and dim==3) to HTML tables """
        if len(a.shape)==2:
            return f'<table style="{style}">' + "".join("<tr>" +
                    "".join(f'<td style="{css_td}">{val:.3g}</td>' for val in row) +
                    "</tr>" for row in a) + "</table>"

        rows = []
        
        if len(a.shape)==3:
            for i in range(a.shape[1]):
                cells = []
                for k in range(a.shape[0]):
                    cells += [f'<td style="{css_td}">{val:.3g}</td>' for val in a[k, i]]
                    if k<a.shape[0]-1:
                        cells.append(f'<td style="{css_lsep}"></td><td style="{css_rsep}"></td>')
                row = "<tr>" + "".join(cells) + "</tr>"
                rows.append(row)        
        elif len(a.shape)==4: 
            rows = []
            for i in range(a.shape[2]):
                cells = []
                for n in range(a.shape[0]):
                    for k in range(a.shape[1]):
                        for j in range(a.shape[3]):
                            val = a[n, k, i, j]
                            cells.append(f'<td style="{css_td}">{val:.3g}</td>')
                            if j<a.shape[3]-1:
                                cells.append(f'<td style="{css_lsep}"></td><td style="{css_rsep}"></td>')
                        if k<a.shape[1]-1:
                            cells.append(f'<td style="{css_ldsep}"></td><td style="{css_rsep}"></td>')
                    if n<a.shape[0]-1:
                        cells.append(f'<td style="{css_lddsep}"></td><td style="{css_rsep}"></td>')
                row = "<tr>" + "".join(cells) + "</tr>"
                rows.append(row)       
        elif len(a.shape)==5: 
            rows = []
            for i in range(a.shape[3]):
                cells = []
                for n in range(a.shape[0]):
                    for l in range(a.shape[1]):
                        for k in range(a.shape[2]):
                            for j in range(a.shape[4]):
                                val=a[n, l, k, i, j]
                                cells.append(f'<td style="{css_td}">{val:.3g}</td>')
                                if j<a.shape[4]-1:
                                    cells.append(f'<td style="{css_lsep}"></td><td style="{css_rsep}"></td>')
                            if k<a.shape[2]-1:
                                cells.append(f'<td style="{css_ldsep}"></td><td style="{css_rsep}"></td>')
                        if l<a.shape[1]-1:
                            cells.append(f'<td style="{css_lddsep}"></td><td style="{css_rsep}"></td>')
                    if n<a.shape[0]-1:
                        cells.append(f'<td style="{css_ldddsep}"></td><td style="{css_rsep}"></td>')
                row = "<tr>" + "".join(cells) + "</tr>"
                rows.append(row) 
        else:
            cells = [f'<td style="{style}">{str(a)}</td>']
            row = "<tr>" + "".join(cells) + "</tr>"
            rows.append(row)         
        return (
            f'<table style="{style}">'
            + "".join(rows)
            + "</table>"
        )


    def val2html(val):
        if isinstance(val, Markup):
            return ''.join([f'<center style="{style};display:block"><span>{val2html(a)}</span></center>' for i, a in enumerate(val.args)])
        elif isinstance(val, np.ndarray):
            return np2html(val)
        elif isinstance(val, tuple):
            for j in range(0,len(val)):
                if isinstance(val[j], np.ndarray):
                    return f'<table style="{style}"><tr>' + ''.join([ f'<td style="padding:0px;">{val2html(val[i])}</td>'+('<td style="style="padding:0px;background-color:none">|</td>' if i<len(val)-1 else '') for i in range(0,len(val))]) + '</tr></table>'
            else:                
                return f'<span style="{style}">' + str(val) + '</span>'
        else:
            if isinstance(val, float):
                val = f'{val:.3f}'
                
            return f'<span style="{style}">' + str(val) + '</span>'

    sep = f'<td style="width:{sep_pad};padding:0px"></td>'
    html = f'<table style="{style}"><tr style="">'+''.join([f'<td  style="vertical-align: top">{val2html(a)}</td>' + (sep if i<len(args)-1 else '') for i, a in enumerate(args)])+'</tr></table>'
    return html
    #for k,v in kwargs.iteritems():
    #print "%s = %s" % (k, v)

def xdisplay(*args, **kwargs):
    display(HTML(html(*args, **kwargs)))    
