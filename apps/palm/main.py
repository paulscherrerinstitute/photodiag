import os
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Range1d, Spacer, Plot, \
    LinearAxis, DataRange1d, Line, CustomJS, MultiLine, Circle
from bokeh.models.annotations import Title
from bokeh.models.grids import Grid
from bokeh.models.tickers import BasicTicker
from bokeh.models.tools import PanTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
from bokeh.models.widgets import Button, Toggle, Panel, Tabs, Dropdown, Select, RadioButtonGroup, TextInput, \
    DataTable, TableColumn
from tornado import gen
from photon_diag.palm_code import PalmSetup

import receiver

doc = curdoc()
doc.title = "PALM"

current_message = None

connected = False

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
WAVEFORM_CANVAS_WIDTH = 1000
WAVEFORM_CANVAS_HEIGHT = 300

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

palm = PalmSetup('')


# Calibration averaged waveforms per photon energy
calib_wf_plot = Plot(
    title=Title(text="Calibration waveforms"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
calib_wf_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
calib_wf_plot.add_layout(LinearAxis(), place='below')
calib_wf_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_wf_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_wf_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
calib_waveform_source = ColumnDataSource(dict(x=[0], y=[0]))
calib_wf_plot.add_glyph(calib_waveform_source, MultiLine(xs='x', ys='y', line_color='red'))


# Calibration fit plot
calib_fit_plot = Plot(
    title=Title(text="Calibration fit"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
calib_fit_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
calib_fit_plot.add_layout(LinearAxis(axis_label='Spectrometer Peak Position, pix'), place='below')
calib_fit_plot.add_layout(LinearAxis(axis_label='Photon Energy, eV', major_label_orientation='vertical'),
                          place='left')

# ---- grid lines
calib_fit_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_fit_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- calibration points circle glyphs
calib_point_source0 = ColumnDataSource(dict(x=[], y=[]))
calib_fit_plot.add_glyph(calib_point_source0, Circle(x='x', y='y', line_color='blue'))
calib_point_source1 = ColumnDataSource(dict(x=[], y=[]))
calib_fit_plot.add_glyph(calib_point_source1, Circle(x='x', y='y', line_color='red'))

# ---- calibration fit line glyphs
calib_fit_source0 = ColumnDataSource(dict(x=[], y=[]))
calib_fit_plot.add_glyph(calib_fit_source0, Line(x='x', y='y', line_color='blue'))
calib_fit_source1 = ColumnDataSource(dict(x=[], y=[]))
calib_fit_plot.add_glyph(calib_fit_source1, Line(x='x', y='y', line_color='red'))


# Streaked and unstreaked waveforms plot
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
waveform_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
waveform_plot.add_layout(LinearAxis(), place='below')
waveform_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- grid lines
waveform_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
waveform_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
waveform_source = ColumnDataSource(dict(x_streaked=[0], y_streaked=[0], x_unstreaked=[0], y_unstreaked=[0]))
waveform_plot.add_glyph(waveform_source, Line(x='x_streaked', y='y_streaked', line_color='red'))
waveform_plot.add_glyph(waveform_source, Line(x='x_unstreaked', y='y_unstreaked', line_color='blue'))


# Streaked and unstreaked waveforms plot
energy_plot = Plot(
    title=Title(text="FEL energies"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
energy_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
energy_plot.add_layout(LinearAxis(), place='below')
energy_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- grid lines
energy_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
energy_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
energy_source = ColumnDataSource(dict(time=[0], undulator=[0], monochrom=[0]))
energy_plot.add_glyph(energy_source, Line(x='time', y='undulator', line_color='red'))
energy_plot.add_glyph(energy_source, Line(x='time', y='monochrom', line_color='blue'))


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global stream_t
    stream_t = 1  # keep the latest point in order to prevent full axis reset
    energy_source.data.update(time=[1], undulator=[energy_source.data['undulator'][-1]],
                              monochrom=[energy_source.data['monochrom'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


# Calibration panel
def calibration_path_update():
    new_menu = [('None', 'None')]
    if os.path.isdir(calibration_path.value):
        with os.scandir(calibration_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))

    background_dropdown.menu = sorted(new_menu)

doc.add_periodic_callback(calibration_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)


# ---- calibration folder path text input
def calibration_path_callback(attr, old, new):
    calibration_path_update()

calibration_path = TextInput(title="Calibration Folder Path:", value=HDF5_FILE_PATH, width=250)
calibration_path.on_change('value', calibration_path_callback)


# ---- background dropdown menu
def background_dropdown_callback(selection):
    background_dropdown.label = f"Background energy: {selection}"

background_dropdown = Dropdown(label="Background energy: None", button_type='primary', menu=[], width=250)
background_dropdown.on_click(background_dropdown_callback)


# ---- load button
def calibrate_button_callback():
    def plot_fit(time, calib_a, calib_b):
        time_fit = np.linspace(time.min(), time.max(), 100)
        en_fit = (calib_a / time_fit) ** 2 + calib_b
        return time_fit, en_fit

    def update_calib_plot(calib_results, circle, line):
        (a, c), x, y = calib_results
        x_fit, y_fit = plot_fit(x, a, c)
        circle.data.update(x=x, y=y)
        line.data.update(x=x_fit, y=y_fit)

    calib_res = palm.calibrate(folder_name=calibration_path.value)
    update_calib_plot(calib_res['0'], calib_point_source0, calib_fit_source0)
    update_calib_plot(calib_res['1'], calib_point_source1, calib_fit_source1)


calibrate_button = Button(label="Calibrate", button_type='default', width=250)
calibrate_button.on_click(calibrate_button_callback)


# assemble
tab_calibration = Panel(
    child=column(calibration_path, background_dropdown, calibrate_button),
    title="Calibration")


# Stream panel
# ---- image buffer slider
def buffer_slider_callback(attr, old, new):
    message = receiver.data_buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, message=message))

buffer_slider_source = ColumnDataSource(dict(value=[]))
buffer_slider_source.on_change('data', buffer_slider_callback)

buffer_slider = Slider(start=0, end=1, value=0, step=1, title="Buffered Image",
                       callback_policy='mouseup')

buffer_slider.callback = CustomJS(
    args=dict(source=buffer_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")


# ---- connect toggle button
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


stream_button = Toggle(label="Connect", button_type='default', width=250)
stream_button.on_click(stream_button_callback)


# assemble
tab_stream = Panel(child=column(buffer_slider, stream_button),
                   title="Stream")


# HDF5 File panel
def hdf5_file_path_update():
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))

    saved_runs_dropdown.menu = sorted(new_menu)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)


# ---- folder path text input
def hdf5_file_path_callback(attr, old, new):
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH, width=250)
hdf5_file_path.on_change('value', hdf5_file_path_callback)


# ---- saved runs dropdown menu
def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[], width=250)
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- dataset path text input
hdf5_dataset_path = TextInput(title="Dataset Path:", value=HDF5_DATASET_PATH, width=250)


# ---- load button
def mx_image(file, dataset, i):
    import hdf5plugin  # required to be loaded prior to h5py
    import h5py
    with h5py.File(file, 'r') as f:
        return f[dataset][i, :].astype(np.float32)


def load_file_button_callback():
    global hdf5_file_data, current_message
    file_name = os.path.join(hdf5_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name, dataset=hdf5_dataset_path.value)
    current_message = hdf5_file_data(i=hdf5_pulse_slider.value)
    update(current_message)

load_file_button = Button(label="Load", button_type='default', width=250)
load_file_button.on_click(load_file_button_callback)


# ---- pulse number slider
def hdf5_pulse_slider_callback(attr, old, new):
    global hdf5_file_data, current_message
    current_message = hdf5_file_data(i=new['value'][0])
    update(current_message)

hdf5_pulse_slider_source = ColumnDataSource(dict(value=[]))
hdf5_pulse_slider_source.on_change('data', hdf5_pulse_slider_callback)

hdf5_pulse_slider = Slider(start=0, end=99, value=0, step=1, title="Pulse Number",
                           callback_policy='mouseup')

hdf5_pulse_slider.callback = CustomJS(
    args=dict(source=hdf5_pulse_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")


# assemble
tab_hdf5file = Panel(
    child=column(hdf5_file_path, saved_runs_dropdown, hdf5_dataset_path, load_file_button, hdf5_pulse_slider),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_calibration, tab_stream, tab_hdf5file])

# Final layouts
layout_main = column(row(calib_wf_plot, calib_fit_plot),
                     row(waveform_plot, energy_plot),
                     intensity_stream_reset_button)
final_layout = row(layout_main, data_source_tabs)

doc.add_root(final_layout)


@gen.coroutine
def update(message):
    global stream_t
    if connected and receiver.state == 'receiving':
        stream_t += 1

        y_unstreaked = message[receiver.unstreaked].value
        x_unstreaked = np.arange(len(y_unstreaked))

        y_streaked = message[receiver.streaked].value
        x_streaked = np.arange(len(y_streaked))

        undulator = message[receiver.undulator].value
        monochrom = message[receiver.monochrom].value

        waveform_source.stream(new_data=dict(x_streaked=[x_streaked], y_streaked=[y_streaked],
                                             x_unstreaked=[x_unstreaked], y_unstreaked=[y_unstreaked]),
                               rollover=STREAM_ROLLOVER)

        energy_source.stream(new_data=dict(time=[stream_t], undulator=[undulator], monochrom=[monochrom]),
                             rollover=STREAM_ROLLOVER)


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

            if len(receiver.data_buffer) > 0:
                current_message = receiver.data_buffer[-1]

    doc.add_next_tick_callback(partial(update, message=current_message))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
