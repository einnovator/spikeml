
import ipywidgets as widgets
from typing import Any, Callable, Optional
from pydantic import BaseModel


def ui(
    model: BaseModel,
    callback_start: Optional[Callable[[BaseModel], None]] = None,
    callback_pause: Optional[Callable[[BaseModel], None]] = None,
    callback_stop: Optional[Callable[[BaseModel], None]] = None,
    callback_change: Optional[Callable[[str, Any, dict], None]] = None,
) -> BaseModel:
    """
    Create an interactive Jupyter UI for inspecting and modifying parameters
    of a Pydantic model using ipywidgets, with start/pause/stop controls
    and optional user-defined callbacks.

    Parameters
    ----------
    model : BaseModel
        The Pydantic model instance whose fields will be visualized and controlled
        by widgets. Each field’s type determines the kind of widget created.
    callback_start : Callable[[BaseModel], None], optional
        Function to be called when the "Start" button is clicked.
    callback_pause : Callable[[BaseModel], None], optional
        Function to be called when the "Pause" button is clicked.
    callback_stop : Callable[[BaseModel], None], optional
        Function to be called when the "Stop" button is clicked.
    callback_change : Callable[[str, Any, dict], None], optional
        Function to be called when any widget value changes. Receives
        `(field_name, new_value, change_dict)` as arguments.

    Returns
    -------
    BaseModel
        The input model (potentially modified by widget interactions).

    Notes
    -----
    - Each numeric, text, or color field in the model is mapped to an appropriate widget:
        * `float` → FloatSlider + FloatText (linked)
        * `int` → IntSlider + IntText (linked)
        * `str` → Text input
        * custom `'color'` type → ColorPicker
    - The UI automatically wires up widget change callbacks.
    - Buttons for Start/Pause/Stop control are displayed above the parameter widgets.
    - Requires execution inside a Jupyter Notebook environment.
    """
    btn_style = ''
    btn_start = widgets.Button(value=False, description='Start', disabled=False, button_style=btn_style, tooltip='Start', icon='play')
    btn_pause = widgets.Button(value=False, description='Pause', disabled=False, button_style=btn_style, tooltip='Pause', icon='pause')
    btn_stop = widgets.Button(value=False, description='Stop', disabled=False, button_style=btn_style, tooltip='Stop', icon='stop')
    tb = widgets.HBox([btn_start, btn_pause, btn_stop])
    display(tb)
    
      
    def create_widget(model, key, field):
        _type = field.annotation
        value = field.default
        w = None
        if _type==float:
            w1 = widgets.FloatSlider(description=f'{name}', value=value, step=.1) #, min=None, max=None
            w2 = widgets.FloatText(description=f'{name}', value=value) #min=None, max=None
            link = widgets.jslink((w1, 'value'), (w2, 'value'))
            w = widgets.HBox([w1, w2])
            def _on_change(change):
                print(key, w1.value, change)
                if callback_change is not None:
                    callback_change(key, w1.value, change)
            w1.observe(_on_change, names='value')
        elif _type==int:
            w = widgets.IntSlider(description=f'{name}', value=value) #min=None, max=None
            w1 = widgets.IntSlider(description=f'{name}', value=value, step=1) #, min=None, max=None
            w2 = widgets.IntText(description=f'{name}', value=value) #min=None, max=None
            link = widgets.jslink((w1, 'value'), (w2, 'value'))
            w = widgets.HBox([w1, w2])
            def _on_change(change):
                if callback_change is not None:
                    callback_change(key, w1.value, change)
            w1.observe(_on_change, names='value')
        elif _type==str:
            w = widgets.Text(description=f'{name}', value=value, disabled=False)
            def _on_change(change):
                if callback_change is not None:
                    callback_change(key, w.value, change)
            w.observe(_on_change, names='value')
        elif _type.__name__=='color':
            w = widgets.ColorPicker(concise=True, description=f'{name}', value=value, disabled=False)
            def _on_change(change):
                if callback_change is not None:
                    callback_change(key, w.value, change)
            w.observe(_on_change, names='value')
        return w


    fields = model.__class__.model_fields
    
    ww = []
    for name, field in fields.items():
        w = create_widget(model, name, field)
        #print(name, field, w)
        if w is not None:
            ww.append(w)

    tb = widgets.VBox(ww)
    display(tb)
    
    output = widgets.Output()

    def on_change(change):
        output.clear_output()
        with output:
            draw(n_slider.value, pk_color.value)

    def on_btn_start(b):
        output.clear_output()
        with output:
            print("Start!")
            if callback_start is not None:
                callback_start(model)

    def on_btn_pause(b):
        output.clear_output()
        with output:
            print("Pause!")
            if callback_pause is not None:
                callback_pause(model)

    def on_btn_stop(b):
        output.clear_output()
        with output:
            print("Stop")
            if callback_stop is not None:
                callback_stop(model)
            
    btn_start.on_click(on_btn_start)
    btn_pause.on_click(on_btn_pause)
    btn_stop.on_click(on_btn_stop)

    #play = widgets.Play(value=100, min=100, max=1000, step=100, interval=500, description="Play", disabled=False)
    #widgets.jslink((play, 'value'), (n_slider, 'value'))
    #display(widgets.HBox([play]))
    
    display(output)
    return params
