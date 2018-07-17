import os
from functools import partial

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, CheckboxButtonGroup, \
    Circle, ColumnDataSource, CustomJS, DataRange1d, Div, Dropdown, Grid, \
    HoverTool, Legend, Line, LinearAxis, MultiLine, Panel, PanTool, Plot, \
    ResetTool, Slider, Spacer, Span, Tabs, TextInput, Title, Toggle, WheelZoomTool
from tornado import gen

import photodiag
import receiver

doc = curdoc()
doc.title = "PALM"

current_message = None
current_results = ()

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
WAVEFORM_CANVAS_WIDTH = 700
WAVEFORM_CANVAS_HEIGHT = 400

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600

DEFAULT_CALIB_PATH = os.path.join(os.path.expanduser('~'), 'eTOF_calibs')
HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

palm = photodiag.PalmSetup({'0': receiver.reference, '1': receiver.streaked})


# Calibration averaged waveforms per photon energy
calib_wf_plot = Plot(
    title=Title(text="eTOF calibration waveforms"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
calib_wf_plot_hover = HoverTool(tooltips=[
    ("energy, eV", '@en'),
    ("flight time, ns", '$x'),
    ("intensity, a.u.", '$y'),
])

calib_wf_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), calib_wf_plot_hover)

# ---- axes
calib_wf_plot.add_layout(LinearAxis(axis_label='Spectrometer internal time'), place='below')
calib_wf_plot.add_layout(
    LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_wf_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_wf_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- multiline calibration waveforms glyphs
calib_waveform_source0 = ColumnDataSource(dict(xs=[], ys=[], en=[]))
reference_ml = calib_wf_plot.add_glyph(
    calib_waveform_source0, MultiLine(xs='xs', ys='ys', line_color='blue'))

calib_waveform_source1 = ColumnDataSource(dict(xs=[], ys=[], en=[]))
streaked_ml = calib_wf_plot.add_glyph(
    calib_waveform_source1, MultiLine(xs='xs', ys='ys', line_color='red'))

calib_wf_plot.add_layout(Legend(items=[
    ("reference", [reference_ml]),
    ("streaked", [streaked_ml])
]))
calib_wf_plot.legend.click_policy = "hide"

# ---- photon peak position
phot_peak_pos_ref = Span(location=0, dimension='height', line_dash='dashed', line_color='blue')
phot_peak_pos_str = Span(location=0, dimension='height', line_dash='dashed', line_color='red')
calib_wf_plot.add_layout(phot_peak_pos_ref)
calib_wf_plot.add_layout(phot_peak_pos_str)


# Calibration fit plot
calib_fit_plot = Plot(
    title=Title(text="eTOF calibration fit"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
calib_fit_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

# ---- axes
calib_fit_plot.add_layout(LinearAxis(axis_label='Photoelectron peak shift'), place='below')
calib_fit_plot.add_layout(
    LinearAxis(axis_label='X-fel energy, eV', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_fit_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_fit_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- calibration fit points circle glyphs
calib_point_source0 = ColumnDataSource(dict(x=[], y=[]))
reference_c = calib_fit_plot.add_glyph(calib_point_source0, Circle(x='x', y='y', line_color='blue'))
calib_point_source1 = ColumnDataSource(dict(x=[], y=[]))
streaked_c = calib_fit_plot.add_glyph(calib_point_source1, Circle(x='x', y='y', line_color='red'))

# ---- calibration fit line glyphs
calib_fit_source0 = ColumnDataSource(dict(x=[], y=[]))
reference_l = calib_fit_plot.add_glyph(calib_fit_source0, Line(x='x', y='y', line_color='blue'))
calib_fit_source1 = ColumnDataSource(dict(x=[], y=[]))
streaked_l = calib_fit_plot.add_glyph(calib_fit_source1, Line(x='x', y='y', line_color='red'))

calib_fit_plot.add_layout(Legend(items=[
    ("reference", [reference_c, reference_l]),
    ("streaked", [streaked_c, streaked_l])
]))
calib_fit_plot.legend.click_policy = "hide"


# THz calibration plot
calib_thz_plot = Plot(
    title=Title(text="THz calibration"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=WAVEFORM_CANVAS_HEIGHT,
    plot_width=WAVEFORM_CANVAS_WIDTH,
    toolbar_location='right',
    logo=None,
)

# ---- tools
calib_thz_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

# ---- axes
calib_thz_plot.add_layout(LinearAxis(axis_label='Stage delay position'), place='below')
calib_thz_plot.add_layout(
    LinearAxis(axis_label='Energy shift, eV', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_thz_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_thz_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- calibration fit points circle glyphs
calib_thz_point_source = ColumnDataSource(dict(x=[], y=[]))
thz_fit_c = calib_thz_plot.add_glyph(
    calib_thz_point_source, Circle(x='x', y='y', line_color='blue'))

# ---- calibration fit line glyphs
calib_thz_fit_source = ColumnDataSource(dict(x=[], y=[]))
thz_fit_l = calib_thz_plot.add_glyph(calib_thz_fit_source, Line(x='x', y='y', line_color='blue'))


# Streaked and reference waveforms plot
waveform_plot = Plot(
    title=Title(text="Waveforms"),
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
waveform_plot.add_layout(LinearAxis(axis_label='Photon energy, eV'), place='below')
waveform_plot.add_layout(
    LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

# ---- grid lines
waveform_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
waveform_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- waveforms line glyphs
waveform_source = ColumnDataSource(dict(x_str=[], y_str=[], x_ref=[], y_ref=[]))
reference_l = waveform_plot.add_glyph(
    waveform_source, Line(x='x_ref', y='y_ref', line_color='blue'))
streaked_l = waveform_plot.add_glyph(
    waveform_source, Line(x='x_str', y='y_str', line_color='red'))

waveform_plot.add_layout(Legend(items=[
    ("reference", [reference_l]),
    ("streaked", [streaked_l])
]))
waveform_plot.legend.click_policy = "hide"


# Cross-correlation plot
xcorr_plot = Plot(
    title=Title(text="Cross-correlation"),
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
xcorr_plot.add_layout(LinearAxis(axis_label='Delay, eV'), place='below')
xcorr_plot.add_layout(
    LinearAxis(axis_label='Xcorr, a.u.', major_label_orientation='vertical'), place='left')

# ---- grid lines
xcorr_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
xcorr_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
xcorr_source = ColumnDataSource(dict(lags=[], xcorr1=[], xcorr2=[]))
xcorr_pos_source = ColumnDataSource(dict(pos=[]))
xcorr_plot.add_glyph(
    xcorr_source, Line(x='lags', y='xcorr1', line_color='purple', line_dash='dashed'))
xcorr_plot.add_glyph(xcorr_source, Line(x='lags', y='xcorr2', line_color='purple'))
xcorr_plot_pos = Span(location=0, dimension='height')
xcorr_plot.add_layout(xcorr_plot_pos)

# Delays plot
delay_plot = Plot(
    title=Title(text="Delays"),
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
delay_plot.add_layout(LinearAxis(axis_label='Shot number'), place='below')
delay_plot.add_layout(
    LinearAxis(axis_label='Delay, eV', major_label_orientation='vertical'), place='left')

# ---- grid lines
delay_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
delay_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
delay_source = ColumnDataSource(dict(pulse=[], delay=[]))
pulse_pos_source = ColumnDataSource(dict(pos=[]))
delay_plot.add_glyph(delay_source, Line(x='pulse', y='delay', line_color='steelblue'))
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
pulse_len_plot.add_layout(LinearAxis(axis_label='Shot number'), place='below')
pulse_len_plot.add_layout(
    LinearAxis(axis_label='Pulse length', major_label_orientation='vertical'), place='left')

# ---- grid lines
pulse_len_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
pulse_len_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
pulse_len_source = ColumnDataSource(dict(x=[], y=[]))
pulse_len_plot.add_glyph(pulse_len_source, Line(x='x', y='y', line_color='steelblue'))
pulse_len_plot_pos = Span(location=0, dimension='height')
pulse_len_plot.add_layout(pulse_len_plot_pos)


# Fitting equation
fit_eq_div = Div(text="""Fitting equation:<br><br><img src="/palm/static/5euwuy.gif">""")

# Calibration constants
calib_const_div = Div(text="")

# Calibration panel
def calibration_path_update():
    new_menu = [('None', 'None')]
    if os.path.isdir(calibration_path.value):
        with os.scandir(calibration_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))

    background_dropdown.menu = sorted(new_menu, reverse=True)

doc.add_periodic_callback(calibration_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)

# ---- calibration folder path text input
def calibration_path_callback(_attr, _old, _new):
    calibration_path_update()

calibration_path = TextInput(title="Calibration Folder Path:", value=HDF5_FILE_PATH, width=250)
calibration_path.on_change('value', calibration_path_callback)

# ---- background dropdown menu
def background_dropdown_callback(selection):
    background_dropdown.label = f"Background energy: {selection}"

background_dropdown = Dropdown(label="Background energy: None", menu=[], width=250)
background_dropdown.on_click(background_dropdown_callback)

# ---- calibrate button
def calibrate_button_callback():
    calib_res = palm.calibrate_etof(folder_name=calibration_path.value)
    update_calibration_plot(calib_res)

def update_calibration_plot(calib_res):
    etof_ref = palm.etofs['0']
    etof_str = palm.etofs['1']

    calib_waveform_source0.data.update(
        xs=len(etof_ref.calib_data)*[list(range(etof_ref.internal_time_bins))],
        ys=etof_ref.calib_data['waveform'].tolist(),
        en=etof_ref.calib_data.index.values)

    calib_waveform_source1.data.update(
        xs=len(etof_str.calib_data)*[list(range(etof_str.internal_time_bins))],
        ys=etof_str.calib_data['waveform'].tolist(),
        en=etof_str.calib_data.index.values)

    phot_peak_pos_ref.location = etof_ref.calib_t0
    phot_peak_pos_str.location = etof_str.calib_t0

    def plot_fit(time, calib_a, calib_b):
        time_fit = np.linspace(time.min(), time.max(), 100)
        en_fit = (calib_a / time_fit) ** 2 + calib_b
        return time_fit, en_fit

    def update_plot(calib_results, circle, line):
        (a, c), x, y = calib_results
        x_fit, y_fit = plot_fit(x, a, c)
        circle.data.update(x=x, y=y)
        line.data.update(x=x_fit, y=y_fit)

    update_plot(calib_res['0'], calib_point_source0, calib_fit_source0)
    update_plot(calib_res['1'], calib_point_source1, calib_fit_source1)

    calib_const_div.text = f"""
    a_str = {etof_str.calib_a:.2f}<br>
    b_str = {etof_str.calib_b:.2f}<br>
    <br>
    a_ref = {etof_ref.calib_a:.2f}<br>
    b_ref = {etof_ref.calib_b:.2f}
    """

calibrate_button = Button(label="Calibrate", button_type='default', width=250)
calibrate_button.on_click(calibrate_button_callback)

# ---- save calibration button
def save_button_callback():
    palm.save_etof_calib()
    update_calib_load_menu()

save_button = Button(label="Save", button_type='default', width=120)
save_button.on_click(save_button_callback)

# ---- load calibration button
def load_button_callback(selection):
    calib_file = os.path.join(DEFAULT_CALIB_PATH, selection)
    palm.load_etof_calib(calib_file)
    calib_results = {}
    for etof_key in palm.etofs:
        calib_results[etof_key] = palm.etofs[etof_key].fit_calibration_curve()
    update_calibration_plot(calib_results)

def update_calib_load_menu():
    if os.path.isdir(DEFAULT_CALIB_PATH):
        new_menu = []
        with os.scandir(DEFAULT_CALIB_PATH) as it:
            for entry in it:
                if entry.is_file():
                    new_menu.append((entry.name, entry.name))
        load_button.menu = sorted(new_menu, reverse=True)

doc.add_next_tick_callback(update_calib_load_menu)
doc.add_periodic_callback(update_calib_load_menu, 10000)

load_button = Dropdown(label="Load", menu=[], width=120)
load_button.on_click(load_button_callback)

# assemble
tab_calibration = Panel(
    child=column(
        calibration_path, background_dropdown, calibrate_button,
        row(save_button, Spacer(width=10), load_button)),
    title="Calibration")


# Stream panel
# ---- image buffer slider
def buffer_slider_callback(_attr, _old, new):
    message = receiver.data_buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, message=message))

buffer_slider_source = ColumnDataSource(dict(value=[]))
buffer_slider_source.on_change('data', buffer_slider_callback)

buffer_slider = Slider(
    start=0, end=1, value=0, step=1, title="Buffered Image", callback_policy='mouseup')

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

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)

# assemble
tab_stream = Panel(
    child=column(buffer_slider, stream_button, intensity_stream_reset_button), title="Stream")


# HDF5 File panel
def hdf5_file_path_update():
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))
    saved_runs_dropdown.menu = sorted(new_menu, reverse=True)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)


# ---- folder path text input
def hdf5_file_path_callback(_attr, _old, new):
    save_ti.value = new
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH, width=250)
hdf5_file_path.on_change('value', hdf5_file_path_callback)


# ---- saved runs dropdown menu
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

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[], width=250)
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- save location
save_ti = TextInput(title="Save Folder Path:", value=HDF5_FILE_PATH, width=250)

# ---- autosave checkbox
autosave_cb = CheckboxButtonGroup(labels=["Auto Save"], active=[], width=100)

# ---- save button
def save_b_callback():
    if current_results:
        filename, tags, delays, pulse_lengths = current_results
        save_filename = os.path.splitext(filename)[0]+'.csv'
        df = pd.DataFrame({'pulse_id': tags, 'pulse_delay': delays, 'pulse_length': pulse_lengths})
        df.to_csv(os.path.join(save_ti.value, save_filename), index=False)

save_b = Button(label="Save Results", button_type='default', width=250)
save_b.on_click(save_b_callback)

# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_update_fun
    hdf5_update_fun(pulse=new)

hdf5_pulse_slider = Slider(start=0, end=99999, value=0, step=1, title="Pulse ID")
hdf5_pulse_slider.on_change('value', hdf5_pulse_slider_callback)


# assemble
tab_hdf5file = Panel(
    child=column(
        hdf5_file_path, saved_runs_dropdown, hdf5_pulse_slider, save_ti, autosave_cb, save_b),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_calibration, tab_hdf5file, tab_stream])

# Final layouts
layout_calib = row(
    calib_wf_plot, Spacer(width=50), calib_fit_plot, Spacer(width=50), calib_thz_plot)
layout_proc = row(waveform_plot, Spacer(width=50), xcorr_plot)
layout_res = row(delay_plot, Spacer(width=50), pulse_len_plot)
layout_fit_res = column(fit_eq_div, calib_const_div)
final_layout = column(
    row(layout_calib),
    row(layout_proc, Spacer(width=30), data_source_tabs, Spacer(width=30), layout_fit_res),
    row(layout_res, Spacer(width=30)))

doc.add_root(final_layout)


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
