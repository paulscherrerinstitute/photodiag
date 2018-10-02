from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, CustomJS, Panel, Slider, Tabs, Toggle
from tornado import gen

import receiver
from common import palm
from panel_calib import tab_calibration
from panel_h5file import tab_hdf5file, waveform_plot, waveform_source

doc = curdoc()
doc.title = "PALM"

current_message = None

connected = False

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600


# Stream panel
# Image buffer slider
def buffer_slider_callback(_attr, _old, new):
    message = receiver.data_buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, message=message))

buffer_slider_source = ColumnDataSource(dict(value=[]))
buffer_slider_source.on_change('data', buffer_slider_callback)

buffer_slider = Slider(
    start=0, end=1, value=0, step=1, title="Buffered Image", callback_policy='mouseup',
    disabled=True)

buffer_slider.callback = CustomJS(
    args=dict(source=buffer_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")


# Connect toggle button
def stream_button_callback(state):
    global connected
    if state:
        connected = True
        stream_button.label = 'Connecting'
        stream_button.button_type = 'default'

    else:
        connected = False
        stream_button.label = 'Connect'
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect", button_type='default', disabled=True)
stream_button.on_click(stream_button_callback)


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global stream_t
    stream_t = 1  # keep the latest point in order to prevent full axis reset

intensity_stream_reset_button = Button(label="Reset", button_type='default', disabled=True)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


# Stream update coroutines
@gen.coroutine
def update(message):
    if connected and receiver.state == 'receiving':
        y_ref = message[receiver.reference].value[np.newaxis, :]
        y_str = message[receiver.streaked].value[np.newaxis, :]

        y_ref = palm.etofs['0'].convert(y_ref, palm.energy_range)
        y_str = palm.etofs['1'].convert(y_str, palm.energy_range)

        waveform_source.data.update(
            x_str=palm.energy_range, y_str=y_str[0, :],
            x_ref=palm.energy_range, y_ref=y_ref[0, :])

@gen.coroutine
def internal_periodic_callback():
    global current_message
    if waveform_plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            # Set slider to the right-most position
            if len(receiver.data_buffer) > 1:
                buffer_slider.end = len(receiver.data_buffer) - 1
                buffer_slider.value = len(receiver.data_buffer) - 1

            if receiver.data_buffer:
                current_message = receiver.data_buffer[-1]

    doc.add_next_tick_callback(partial(update, message=current_message))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)


# assemble
tab_stream_layout = column(buffer_slider, stream_button, intensity_stream_reset_button)

tab_stream = Panel(child=tab_stream_layout, title="Stream")


# FINAL LAYOUT
doc.add_root(Tabs(tabs=[tab_calibration, tab_hdf5file, tab_stream]))
