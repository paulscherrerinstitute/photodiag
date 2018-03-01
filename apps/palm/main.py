import os
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Range1d, Spacer, Plot, Legend, \
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
WAVEFORM_CANVAS_WIDTH = 730
WAVEFORM_CANVAS_HEIGHT = 400

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
calib_wf_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
calib_wf_plot.add_layout(LinearAxis(axis_label='Spectrometer internal time, pix'), place='below')
calib_wf_plot.add_layout(LinearAxis(axis_label='Intensity', major_label_orientation='vertical'),
                         place='left')

# ---- grid lines
calib_wf_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_wf_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- multiline calibration waveforms glyphs
calib_waveform_source0 = ColumnDataSource(dict(xs=[], ys=[]))
calib_wf_plot.add_glyph(calib_waveform_source0, MultiLine(xs='xs', ys='ys', line_color='blue'))
calib_waveform_source1 = ColumnDataSource(dict(xs=[], ys=[]))
calib_wf_plot.add_glyph(calib_waveform_source1, MultiLine(xs='xs', ys='ys', line_color='red'))

calib_wf_plot.add_layout(Legend(items=[
    ("unstreaked", [calib_wf_plot.renderers[5]]),
    ("streaked", [calib_wf_plot.renderers[6]])
]))
calib_wf_plot.legend.click_policy = "hide"

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
calib_fit_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
calib_fit_plot.add_layout(LinearAxis(axis_label='Spectrometer Peak Shift, pix'), place='below')
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

calib_fit_plot.add_layout(Legend(items=[
    ("unstreaked", [calib_fit_plot.renderers[5], calib_fit_plot.renderers[7]]),
    ("streaked", [calib_fit_plot.renderers[6], calib_fit_plot.renderers[8]])
]))
calib_fit_plot.legend.click_policy = "hide"


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
waveform_plot.add_layout(LinearAxis(axis_label='Photon Energy, eV'), place='below')
waveform_plot.add_layout(LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

# ---- grid lines
waveform_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
waveform_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
waveform_source = ColumnDataSource(dict(x_str=[], y_str=[], x_unstr=[], y_unstr=[]))
waveform_plot.add_glyph(waveform_source, Line(x='x_unstr', y='y_unstr', line_color='blue'))
waveform_plot.add_glyph(waveform_source, Line(x='x_str', y='y_str', line_color='red'))

waveform_plot.add_layout(Legend(items=[
    ("unstreaked", [waveform_plot.renderers[4]]),
    ("streaked", [waveform_plot.renderers[5]])
]))
waveform_plot.legend.click_policy = "hide"

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
energy_source = ColumnDataSource(dict(time=[], undulator=[], monochrom=[]))
energy_plot.add_glyph(energy_source, Line(x='time', y='undulator', line_color='red'))
energy_plot.add_glyph(energy_source, Line(x='time', y='monochrom', line_color='blue'))


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

background_dropdown = Dropdown(label="Background energy: None", menu=[], width=250)
background_dropdown.on_click(background_dropdown_callback)


# ---- load button
def calibrate_button_callback():
    calib_res = palm.calibrate(folder_name=calibration_path.value)

    calib_data0 = palm.spectrometers['0'].calib_data
    calib_data1 = palm.spectrometers['1'].calib_data
    calib_waveform_source0.data.update(xs=len(calib_data0)*[palm.spectrometers['0'].internal_time],
                                       ys=palm.spectrometers['0'].calib_data['waveform'].tolist())
    calib_waveform_source1.data.update(xs=len(calib_data1)*[palm.spectrometers['1'].internal_time],
                                       ys=palm.spectrometers['1'].calib_data['waveform'].tolist())

    def plot_fit(time, calib_a, calib_b):
        time_fit = np.linspace(time.min(), time.max(), 100)
        en_fit = (calib_a / time_fit) ** 2 + calib_b
        return time_fit, en_fit

    def update_calib_plot(calib_results, circle, line):
        (a, c), x, y = calib_results
        x_fit, y_fit = plot_fit(x, a, c)
        circle.data.update(x=x, y=y)
        line.data.update(x=x_fit, y=y_fit)
        # a_w.value = round(a, 2)
        # c_w.value = round(c)

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


# ---- intensity stream reset button
def intensity_stream_reset_button_callback():
    global stream_t
    stream_t = 1  # keep the latest point in order to prevent full axis reset
    energy_source.data.update(time=[1], undulator=[energy_source.data['undulator'][-1]],
                              monochrom=[energy_source.data['monochrom'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)

# assemble
tab_stream = Panel(child=column(buffer_slider, stream_button, intensity_stream_reset_button),
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
hdf5_update_fun = []
def hdf5_update(pulse, results, prep_data):
    lags, delays, pulse_lengths = results
    waveform_source.data.update(
        x_str=palm.spectrometers['1'].interp_energy, y_str=prep_data['1'][pulse, :],
        x_unstr=palm.spectrometers['0'].interp_energy, y_unstr=prep_data['0'][pulse, :])


def saved_runs_dropdown_callback(selection):
    global hdf5_update_fun
    saved_runs_dropdown.label = selection
    results, prep_data = palm.process_hdf5_file(filename=os.path.join(hdf5_file_path.value, selection))
    hdf5_update_fun = partial(hdf5_update, results=results, prep_data=prep_data)

    hdf5_pulse_slider.end = len(results[1]) - 1
    hdf5_pulse_slider.value = 0
    hdf5_update_fun(0)

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[], width=250)
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)


# ---- pulse number slider
def hdf5_pulse_slider_callback(attr, old, new):
    global hdf5_update_fun
    hdf5_update_fun(pulse=new)

hdf5_pulse_slider = Slider(start=0, end=99, value=0, step=1, title="Pulse ID")
hdf5_pulse_slider.on_change('value', hdf5_pulse_slider_callback)


# assemble
tab_hdf5file = Panel(
    child=column(hdf5_file_path, saved_runs_dropdown, hdf5_pulse_slider),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_calibration, tab_hdf5file, tab_stream])

# Final layouts
layout_main = column(row(calib_wf_plot, Spacer(width=50), calib_fit_plot),
                     row(waveform_plot, Spacer(width=50), energy_plot))
final_layout = row(layout_main, Spacer(width=30), data_source_tabs)

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
