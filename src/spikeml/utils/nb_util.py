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

    def np2html(a):
        """" Convert matrices to HTML tables """
        if len(a.shape)>=2:
            return f'<table style="{style}">' + "".join("<tr>" + "".join(f'<td style="padding:1px;padding-right:2px">{val:.3f}</td>' for val in row) + "</tr>" for row in a) + "</table>"
        else:
            return f'<span style="{style}">' + str(a) + '</span>'

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
