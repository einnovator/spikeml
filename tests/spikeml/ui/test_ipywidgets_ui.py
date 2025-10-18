

from spikelogik.ui.ipywidget_ui import ui

def test_ui():
    params = SSNNParams()
    print(params.fmt())
    fields = params.__class__.model_fields
    for key, field in fields.items():
        print(key, field, field.annotation, field.annotation==float, field.default, field.metadata)
        
    ui(params, callback_start = lambda params: print('START', params), callback_pause = lambda params: print('PAUSE', params), callback_stop = lambda: print('STOP', params), callback_change=lambda name,value,w: print('CHANGE:', name, value))
    #ui(params, callback_start = None, callback_pause = None, callback_stop = None, callback_change= None)
