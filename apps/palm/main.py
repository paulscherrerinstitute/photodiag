import os
from functools import partial

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, Button, CheckboxButtonGroup, ColumnDataSource, \
    CustomJS, DataRange1d, Dropdown, Grid, Legend, Line, LinearAxis, Panel, PanTool, \
    Plot, ResetTool, Slider, Spacer, Span, Tabs, TextInput, Title, Toggle, WheelZoomTool
from tornado import gen

import receiver
from common import WAVEFORM_CANVAS_HEIGHT, WAVEFORM_CANVAS_WIDTH, palm
from panel_calib import tab_calibration

doc = curdoc()
doc.title = "PALM"

current_message = None
current_results = ()

connected = False

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

energy_min = 4850
energy_max = 5150
energy_num_points = 301


# HDF5 FILE PANEL
# Streaked and reference waveforms plot
waveform_plot = Plot(
    title=Title(text="eTOF waveforms"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
waveform_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

# ---- axes
waveform_plot.add_layout(
    LinearAxis(axis_label='Photon energy, eV'), place='below')
waveform_plot.add_layout(
    LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

# ---- grid lines
waveform_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
waveform_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyphs
waveform_source = ColumnDataSource(dict(x_str=[], y_str=[], x_ref=[], y_ref=[]))
reference_l = waveform_plot.add_glyph(
    waveform_source, Line(x='x_ref', y='y_ref', line_color='blue'))
streaked_l = waveform_plot.add_glyph(
    waveform_source, Line(x='x_str', y='y_str', line_color='red'))

# ---- legend
waveform_plot.add_layout(Legend(items=[
    ("reference", [reference_l]),
    ("streaked", [streaked_l])
]))
waveform_plot.legend.click_policy = "hide"


# Cross-correlation plot
xcorr_plot = Plot(
    title=Title(text="Waveforms cross-correlation"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
xcorr_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

# ---- axes
xcorr_plot.add_layout(
    LinearAxis(axis_label='Energy shift, eV'), place='below')
xcorr_plot.add_layout(
    LinearAxis(axis_label='Cross-correlation', major_label_orientation='vertical'), place='left')

# ---- grid lines
xcorr_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
xcorr_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyphs
xcorr_source = ColumnDataSource(dict(lags=[], xcorr1=[], xcorr2=[]))
xcorr_pos_source = ColumnDataSource(dict(pos=[]))
xcorr_plot.add_glyph(
    xcorr_source, Line(x='lags', y='xcorr1', line_color='purple', line_dash='dashed'))
xcorr_plot.add_glyph(xcorr_source, Line(x='lags', y='xcorr2', line_color='purple'))

# ---- vertical span
xcorr_plot_pos = Span(location=0, dimension='height')
xcorr_plot.add_layout(xcorr_plot_pos)


# Delays plot
delay_plot = Plot(
    title=Title(text="Pulse delays"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
delay_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

# ---- axes
delay_plot.add_layout(
    LinearAxis(axis_label='Pulse number'), place='below')
delay_plot.add_layout(
    LinearAxis(axis_label='Pulse delay (uncalib), eV', major_label_orientation='vertical'),
    place='left')

# ---- grid lines
delay_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
delay_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyphs
delay_source = ColumnDataSource(dict(pulse=[], delay=[]))
pulse_pos_source = ColumnDataSource(dict(pos=[]))
delay_plot.add_glyph(delay_source, Line(x='pulse', y='delay', line_color='steelblue'))

# ---- vertical span
delay_plot_pos = Span(location=0, dimension='height')
delay_plot.add_layout(delay_plot_pos)


# Pulse lengths plot
pulse_len_plot = Plot(
    title=Title(text="Pulse lengths"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
pulse_len_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

# ---- axes
pulse_len_plot.add_layout(
    LinearAxis(axis_label='Pulse number'), place='below')
pulse_len_plot.add_layout(
    LinearAxis(axis_label='Pulse length (uncalib), eV', major_label_orientation='vertical'),
    place='left')

# ---- grid lines
pulse_len_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
pulse_len_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyphs
pulse_len_source = ColumnDataSource(dict(x=[], y=[]))
pulse_len_plot.add_glyph(pulse_len_source, Line(x='x', y='y', line_color='steelblue'))

# ---- vertical span
pulse_len_plot_pos = Span(location=0, dimension='height')
pulse_len_plot.add_layout(pulse_len_plot_pos)


# Folder path text input
def hdf5_file_path_callback(_attr, _old, new):
    save_ti.value = new
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH, width=525)
hdf5_file_path.on_change('value', hdf5_file_path_callback)


# Saved runs dropdown menu
hdf5_update_fun = []
def hdf5_update(pulse, delays, debug_data):
    prep_data, lags, corr_res_uncut, corr_results = debug_data

    waveform_source.data.update(
        x_str=palm.energy_range, y_str=prep_data['1'][pulse, :],
        x_ref=palm.energy_range, y_ref=prep_data['0'][pulse, :])

    xcorr_source.data.update(
        lags=lags, xcorr1=corr_res_uncut[pulse, :], xcorr2=corr_results[pulse, :])

    xcorr_plot_pos.location = delays[pulse]
    delay_plot_pos.location = pulse
    pulse_len_plot_pos.location = pulse

def saved_runs_dropdown_callback(selection):
    if selection != "Saved Runs":
        global hdf5_update_fun, current_results
        saved_runs_dropdown.label = selection
        filepath = os.path.join(hdf5_file_path.value, selection)
        tags, delays, pulse_lengths, debug_data = palm.process_hdf5_file(filepath=filepath, debug=True)
        current_results = (selection, tags, delays, pulse_lengths)

        if autosave_cb.active:
            save_b_callback()

        delay_source.data.update(pulse=np.arange(len(delays)), delay=delays)
        pulse_len_source.data.update(x=np.arange(len(pulse_lengths)), y=pulse_lengths)
        hdf5_update_fun = partial(hdf5_update, delays=delays, debug_data=debug_data)

        hdf5_pulse_slider.end = len(delays) - 1
        hdf5_pulse_slider.value = 0
        hdf5_update_fun(0)

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[])
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- saved run periodic update
def hdf5_file_path_update():
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))
    saved_runs_dropdown.menu = sorted(new_menu, reverse=True)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)


# Pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_update_fun
    hdf5_update_fun(pulse=new)

hdf5_pulse_slider = Slider(start=0, end=99999, value=0, step=1, title="Pulse ID", width=500)
hdf5_pulse_slider.on_change('value', hdf5_pulse_slider_callback)


# Energy range and number of interpolation points text inputs
def energy_max_ti_callback(_attr, old, new):
    global energy_max
    try:
        new_value = float(new)
        if new_value > energy_min:
            energy_max = new_value
            palm.energy_range = np.linspace(energy_min, energy_max, energy_num_points)
            saved_runs_dropdown_callback(saved_runs_dropdown.label)
        else:
            energy_max_ti.value = old

    except ValueError:
        energy_max_ti.value = old

def energy_min_ti_callback(_attr, old, new):
    global energy_min
    try:
        new_value = float(new)
        if new_value < energy_max:
            energy_min = new_value
            palm.energy_range = np.linspace(energy_min, energy_max, energy_num_points)
            saved_runs_dropdown_callback(saved_runs_dropdown.label)
        else:
            energy_min_ti.value = old

    except ValueError:
        energy_min_ti.value = old

def energy_num_points_ti_callback(_attr, old, new):
    global energy_num_points
    try:
        new_value = int(new)
        if new_value > 1:
            energy_num_points = new_value
            palm.energy_range = np.linspace(energy_min, energy_max, energy_num_points)
            saved_runs_dropdown_callback(saved_runs_dropdown.label)
        else:
            energy_num_points_ti.value = old

    except ValueError:
        energy_num_points_ti.value = old

energy_max_ti = TextInput(title='Maximal Energy, eV:', value=str(energy_max))
energy_max_ti.on_change('value', energy_max_ti_callback)
energy_min_ti = TextInput(title='Minimal Energy, eV:', value=str(energy_min))
energy_min_ti.on_change('value', energy_min_ti_callback)
energy_num_points_ti = TextInput(
    title='Number of interpolation points:', value=str(energy_num_points))
energy_num_points_ti.on_change('value', energy_num_points_ti_callback)


# Save location
save_ti = TextInput(title="Save Folder Path:", value=HDF5_FILE_PATH, width=525)


# Autosave checkbox
autosave_cb = CheckboxButtonGroup(labels=["Auto Save"], active=[], width=100)


# Save button
def save_b_callback():
    if current_results:
        filename, tags, delays, pulse_lengths = current_results
        save_filename = os.path.splitext(filename)[0] + '.csv'
        df = pd.DataFrame({'pulse_id': tags, 'pulse_delay': delays, 'pulse_length': pulse_lengths})
        df.to_csv(os.path.join(save_ti.value, save_filename), index=False)

save_b = Button(label="Save Results", button_type='default')
save_b.on_click(save_b_callback)


# assemble
tab_hdf5file_layout = column(
    row(
        column(waveform_plot, xcorr_plot), Spacer(width=30),
        column(
            hdf5_file_path, saved_runs_dropdown, hdf5_pulse_slider, Spacer(height=30),
            energy_min_ti, energy_max_ti, energy_num_points_ti, Spacer(height=30),
            save_ti, autosave_cb, save_b)),
    row(delay_plot, Spacer(width=10), pulse_len_plot))

tab_hdf5file = Panel(child=tab_hdf5file_layout, title="HDF5 File")


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
