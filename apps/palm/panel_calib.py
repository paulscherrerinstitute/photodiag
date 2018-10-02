import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, Circle, ColumnDataSource, DataRange1d, \
    DataTable, Div, Dropdown, Grid, HoverTool, IntEditor, Legend, Line, LinearAxis, MultiLine, \
    Panel, PanTool, Plot, ResetTool, Spacer, Span, TableColumn, TextInput, Title, WheelZoomTool

from common import WAVEFORM_CANVAS_HEIGHT, WAVEFORM_CANVAS_WIDTH, palm

doc = curdoc()

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
    ("etof time, a.u.", '$x'),
    ("intensity, a.u.", '$y'),
])

calib_wf_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), calib_wf_plot_hover)

# ---- axes
calib_wf_plot.add_layout(
    LinearAxis(axis_label='Spectrometer internal time'), place='below')
calib_wf_plot.add_layout(
    LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_wf_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_wf_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- multiline glyphs
calib_waveform_source0 = ColumnDataSource(dict(xs=[], ys=[], en=[]))
reference_ml = calib_wf_plot.add_glyph(
    calib_waveform_source0, MultiLine(xs='xs', ys='ys', line_color='blue'))

calib_waveform_source1 = ColumnDataSource(dict(xs=[], ys=[], en=[]))
streaked_ml = calib_wf_plot.add_glyph(
    calib_waveform_source1, MultiLine(xs='xs', ys='ys', line_color='red'))

# ---- legend
calib_wf_plot.add_layout(Legend(items=[
    ("reference", [reference_ml]),
    ("streaked", [streaked_ml])
]))
calib_wf_plot.legend.click_policy = "hide"

# ---- vertical spans
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
calib_fit_plot.add_layout(
    LinearAxis(axis_label='Photoelectron peak shift'), place='below')
calib_fit_plot.add_layout(
    LinearAxis(axis_label='Photon energy, eV', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_fit_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_fit_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- circle glyphs
calib_point_source0 = ColumnDataSource(dict(x=[], y=[]))
reference_c = calib_fit_plot.add_glyph(calib_point_source0, Circle(x='x', y='y', line_color='blue'))
calib_point_source1 = ColumnDataSource(dict(x=[], y=[]))
streaked_c = calib_fit_plot.add_glyph(calib_point_source1, Circle(x='x', y='y', line_color='red'))

# ---- line glyphs
calib_fit_source0 = ColumnDataSource(dict(x=[], y=[]))
reference_l = calib_fit_plot.add_glyph(calib_fit_source0, Line(x='x', y='y', line_color='blue'))
calib_fit_source1 = ColumnDataSource(dict(x=[], y=[]))
streaked_l = calib_fit_plot.add_glyph(calib_fit_source1, Line(x='x', y='y', line_color='red'))

# ---- legend
calib_fit_plot.add_layout(Legend(items=[
    ("reference", [reference_c, reference_l]),
    ("streaked", [streaked_c, streaked_l])
]))
calib_fit_plot.legend.click_policy = "hide"


# Calibration results datatable
def calibres_table_source_callback(_attr, _old, new):
    for en, ps0, ps1 in zip(new['energy'], new['peak_pos0'], new['peak_pos1']):
        palm.etofs['0'].calib_data.loc[en, 'calib_tpeak'] = (ps0 if ps0 != 'NaN' else np.nan)
        palm.etofs['1'].calib_data.loc[en, 'calib_tpeak'] = (ps1 if ps1 != 'NaN' else np.nan)

    calib_res = {}
    for etof_key in palm.etofs:
        calib_res[etof_key] = palm.etofs[etof_key].fit_calibration_curve()
    update_calibration_plot(calib_res)

calibres_table_source = ColumnDataSource(
    dict(energy=['', '', ''], peak_pos0=['', '', ''], peak_pos1=['', '', '']))
calibres_table_source.on_change('data', calibres_table_source_callback)

calibres_table = DataTable(
    source=calibres_table_source,
    columns=[
        TableColumn(field='energy', title="Photon Energy, eV", editor=IntEditor()),
        TableColumn(field='peak_pos0', title="Reference Peak Position", editor=IntEditor()),
        TableColumn(field='peak_pos1', title="Streaked Peak Position", editor=IntEditor())],
    index_position=None,
    editable=True,
    height=200,
    width=500,
)


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
calib_thz_plot.add_layout(
    LinearAxis(axis_label='Stage delay position'), place='below')
calib_thz_plot.add_layout(
    LinearAxis(axis_label='Energy shift, eV', major_label_orientation='vertical'), place='left')

# ---- grid lines
calib_thz_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
calib_thz_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- circle glyphs
calib_thz_point_source = ColumnDataSource(dict(x=[], y=[]))
thz_fit_c = calib_thz_plot.add_glyph(
    calib_thz_point_source, Circle(x='x', y='y', line_color='blue'))

# ---- line glyphs
calib_thz_fit_source = ColumnDataSource(dict(x=[], y=[]))
thz_fit_l = calib_thz_plot.add_glyph(calib_thz_fit_source, Line(x='x', y='y', line_color='blue'))


# Calibration folder path text input
def calib_path_textinput_callback(_attr, _old, _new):
    update_calib_load_menu()

calib_path_textinput = TextInput(
    title="Calibration Folder Path:", value=os.path.join(os.path.expanduser('~')), width=525)
calib_path_textinput.on_change('value', calib_path_textinput_callback)


# Calibrate button
def calibrate_button_callback():
    palm.calibrate_etof(folder_name=calib_path_textinput.value)

    calibres_table_source.data.update(
        energy=palm.etofs['0'].calib_data.index.tolist(),
        peak_pos0=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
        peak_pos1=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

def update_calibration_plot(calib_res):
    etof_ref = palm.etofs['0']
    etof_str = palm.etofs['1']

    calib_waveform_source0.data.update(
        xs=len(etof_ref.calib_data)*[list(range(etof_ref.internal_time_bins))],
        ys=etof_ref.calib_data['waveform'].tolist(),
        en=etof_ref.calib_data.index.tolist())

    calib_waveform_source1.data.update(
        xs=len(etof_str.calib_data)*[list(range(etof_str.internal_time_bins))],
        ys=etof_str.calib_data['waveform'].tolist(),
        en=etof_str.calib_data.index.tolist())

    phot_peak_pos_ref.location = etof_ref.calib_t0
    phot_peak_pos_str.location = etof_str.calib_t0

    def plot_fit(time, calib_a, calib_b):
        time_fit = np.linspace(np.nanmin(time), np.nanmax(time), 100)
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

calibrate_button = Button(label="Calibrate", button_type='default')
calibrate_button.on_click(calibrate_button_callback)


# Save calibration button
def save_button_callback():
    palm.save_etof_calib(path=calib_path_textinput.value)
    update_calib_load_menu()

save_button = Button(label="Save", button_type='default', width=135)
save_button.on_click(save_button_callback)


# Load calibration button
def load_button_callback(selection):
    if selection:
        palm.load_etof_calib(calib_path_textinput.value, selection)

        calibres_table_source.data.update(
            energy=palm.etofs['0'].calib_data.index.tolist(),
            peak_pos0=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
            peak_pos1=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

        # Drop selection, so that this callback can be triggered again on the same dropdown menu
        # item from the user perspective
        load_button.value = ''

def update_calib_load_menu():
    new_menu = []
    if os.path.isdir(calib_path_textinput.value):
        with os.scandir(calib_path_textinput.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.palm')):
                    new_menu.append((entry.name[:-5], entry.name))
        load_button.button_type = 'default'
        load_button.menu = sorted(new_menu, reverse=True)
    else:
        load_button.button_type = 'danger'
        load_button.menu = new_menu

doc.add_next_tick_callback(update_calib_load_menu)
doc.add_periodic_callback(update_calib_load_menu, 10000)

load_button = Dropdown(label="Load", menu=[], width=135)
load_button.on_click(load_button_callback)


# Fitting equation
fit_eq_div = Div(text="""Fitting equation:<br><br><img src="/palm/static/5euwuy.gif">""")


# Calibration constants
calib_const_div = Div(
    text=f"""
    a_str = {0}<br>
    b_str = {0}<br>
    <br>
    a_ref = {0}<br>
    b_ref = {0}
    """)


# assemble
tab_calibration_layout = row(
    column(calib_wf_plot, calib_fit_plot, calib_thz_plot), Spacer(width=30),
    column(
        calib_path_textinput, calibrate_button, row(save_button, Spacer(width=10), load_button),
        calibres_table, fit_eq_div, calib_const_div))

tab_calibration = Panel(child=tab_calibration_layout, title="Calibration")
