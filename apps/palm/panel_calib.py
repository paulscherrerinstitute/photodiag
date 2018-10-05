import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, Circle, ColumnDataSource, DataRange1d, \
    DataTable, Div, Dropdown, Grid, HoverTool, IntEditor, Legend, Line, LinearAxis, MultiLine, \
    Panel, PanTool, Plot, ResetTool, Spacer, Span, TableColumn, TextInput, Title, WheelZoomTool

PLOT_CANVAS_WIDTH = 620
PLOT_CANVAS_HEIGHT = 380

def create(palm):
    doc = curdoc()

    # Calibration averaged waveforms per photon energy
    waveform_plot = Plot(
        title=Title(text="eTOF calibration waveforms"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    waveform_plot_hovertool = HoverTool(tooltips=[
        ("energy, eV", '@en'),
        ("etof time, a.u.", '$x'),
        ("intensity, a.u.", '$y'),
    ])

    waveform_plot.add_tools(
        PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), waveform_plot_hovertool)

    # ---- axes
    waveform_plot.add_layout(
        LinearAxis(axis_label='Spectrometer internal time'), place='below')
    waveform_plot.add_layout(
        LinearAxis(axis_label='Intensity', major_label_orientation='vertical'), place='left')

    # ---- grid lines
    waveform_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    waveform_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- multiline glyphs
    waveform_ref_source = ColumnDataSource(dict(xs=[], ys=[], en=[]))
    waveform_ref_multiline = waveform_plot.add_glyph(
        waveform_ref_source, MultiLine(xs='xs', ys='ys', line_color='blue'))

    waveform_str_source = ColumnDataSource(dict(xs=[], ys=[], en=[]))
    waveform_str_multiline = waveform_plot.add_glyph(
        waveform_str_source, MultiLine(xs='xs', ys='ys', line_color='red'))

    # ---- legend
    waveform_plot.add_layout(Legend(items=[
        ("reference", [waveform_ref_multiline]),
        ("streaked", [waveform_str_multiline])
    ]))
    waveform_plot.legend.click_policy = "hide"

    # ---- vertical spans
    photon_peak_ref_span = Span(
        location=0, dimension='height', line_dash='dashed', line_color='blue')
    photon_peak_str_span = Span(
        location=0, dimension='height', line_dash='dashed', line_color='red')
    waveform_plot.add_layout(photon_peak_ref_span)
    waveform_plot.add_layout(photon_peak_str_span)


    # Calibration fit plot
    fit_plot = Plot(
        title=Title(text="eTOF calibration fit"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    fit_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    fit_plot.add_layout(
        LinearAxis(axis_label='Photoelectron peak shift'), place='below')
    fit_plot.add_layout(
        LinearAxis(axis_label='Photon energy, eV', major_label_orientation='vertical'),
        place='left')

    # ---- grid lines
    fit_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    fit_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- circle glyphs
    fit_ref_circle_source = ColumnDataSource(dict(x=[], y=[]))
    fit_ref_circle = fit_plot.add_glyph(
        fit_ref_circle_source, Circle(x='x', y='y', line_color='blue'))
    fit_str_circle_source = ColumnDataSource(dict(x=[], y=[]))
    fit_str_circle = fit_plot.add_glyph(
        fit_str_circle_source, Circle(x='x', y='y', line_color='red'))

    # ---- line glyphs
    fit_ref_line_source = ColumnDataSource(dict(x=[], y=[]))
    fit_ref_line = fit_plot.add_glyph(
        fit_ref_line_source, Line(x='x', y='y', line_color='blue'))
    fit_str_line_source = ColumnDataSource(dict(x=[], y=[]))
    fit_str_line = fit_plot.add_glyph(
        fit_str_line_source, Line(x='x', y='y', line_color='red'))

    # ---- legend
    fit_plot.add_layout(Legend(items=[
        ("reference", [fit_ref_circle, fit_ref_line]),
        ("streaked", [fit_str_circle, fit_str_line])
    ]))
    fit_plot.legend.click_policy = "hide"


    # Calibration results datatable
    def datatable_source_callback(_attr, _old, new):
        for en, ps0, ps1 in zip(new['energy'], new['peak_pos_ref'], new['peak_pos_str']):
            palm.etofs['0'].calib_data.loc[en, 'calib_tpeak'] = (ps0 if ps0 != 'NaN' else np.nan)
            palm.etofs['1'].calib_data.loc[en, 'calib_tpeak'] = (ps1 if ps1 != 'NaN' else np.nan)

        calib_res = {}
        for etof_key in palm.etofs:
            calib_res[etof_key] = palm.etofs[etof_key].fit_calibration_curve()
        update_calibration_plot(calib_res)

    datatable_source = ColumnDataSource(
        dict(energy=['', '', ''], peak_pos_ref=['', '', ''], peak_pos_str=['', '', '']))
    datatable_source.on_change('data', datatable_source_callback)

    datatable = DataTable(
        source=datatable_source,
        columns=[
            TableColumn(field='energy', title="Photon Energy, eV", editor=IntEditor()),
            TableColumn(field='peak_pos_ref', title="Reference Peak Position", editor=IntEditor()),
            TableColumn(field='peak_pos_str', title="Streaked Peak Position", editor=IntEditor())],
        index_position=None,
        editable=True,
        height=200,
        width=500,
    )


    # THz calibration plot
    thz_scan_plot = Plot(
        title=Title(text="THz calibration"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    thz_scan_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    thz_scan_plot.add_layout(
        LinearAxis(axis_label='Stage delay position'), place='below')
    thz_scan_plot.add_layout(
        LinearAxis(axis_label='Energy shift, eV', major_label_orientation='vertical'), place='left')

    # ---- grid lines
    thz_scan_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    thz_scan_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- circle glyphs
    thz_scan_circle_source = ColumnDataSource(dict(x=[], y=[]))
    thz_scan_plot.add_glyph(thz_scan_circle_source, Circle(x='x', y='y', line_color='purple'))

    # ---- line glyphs
    thz_fit_line_source = ColumnDataSource(dict(x=[], y=[]))
    thz_scan_plot.add_glyph(thz_fit_line_source, Line(x='x', y='y', line_color='purple'))


    # Calibration folder path text input
    def path_textinput_callback(_attr, _old, _new):
        update_load_dropdown_menu()

    path_textinput = TextInput(
        title="Calibration Folder Path:", value=os.path.join(os.path.expanduser('~')), width=525)
    path_textinput.on_change('value', path_textinput_callback)


    # Calibrate button
    def calibrate_button_callback():
        palm.calibrate_etof(folder_name=path_textinput.value)

        datatable_source.data.update(
            energy=palm.etofs['0'].calib_data.index.tolist(),
            peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
            peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

    def update_calibration_plot(calib_res):
        etof_ref = palm.etofs['0']
        etof_str = palm.etofs['1']

        waveform_ref_source.data.update(
            xs=len(etof_ref.calib_data)*[list(range(etof_ref.internal_time_bins))],
            ys=etof_ref.calib_data['waveform'].tolist(),
            en=etof_ref.calib_data.index.tolist())

        waveform_str_source.data.update(
            xs=len(etof_str.calib_data)*[list(range(etof_str.internal_time_bins))],
            ys=etof_str.calib_data['waveform'].tolist(),
            en=etof_str.calib_data.index.tolist())

        photon_peak_ref_span.location = etof_ref.calib_t0
        photon_peak_str_span.location = etof_str.calib_t0

        def plot_fit(time, calib_a, calib_b):
            time_fit = np.linspace(np.nanmin(time), np.nanmax(time), 100)
            en_fit = (calib_a / time_fit) ** 2 + calib_b
            return time_fit, en_fit

        def update_plot(calib_results, circle, line):
            (a, c), x, y = calib_results
            x_fit, y_fit = plot_fit(x, a, c)
            circle.data.update(x=x, y=y)
            line.data.update(x=x_fit, y=y_fit)

        update_plot(calib_res['0'], fit_ref_circle_source, fit_ref_line_source)
        update_plot(calib_res['1'], fit_str_circle_source, fit_str_line_source)

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
        palm.save_etof_calib(path=path_textinput.value)
        update_load_dropdown_menu()

    save_button = Button(label="Save", button_type='default', width=135)
    save_button.on_click(save_button_callback)


    # Load calibration button
    def load_dropdown_callback(selection):
        if selection:
            palm.load_etof_calib(os.path.join(path_textinput.value, selection))

            datatable_source.data.update(
                energy=palm.etofs['0'].calib_data.index.tolist(),
                peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
                peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

            # Drop selection, so that this callback can be triggered again on the same dropdown menu
            # item from the user perspective
            load_dropdown.value = ''

    def update_load_dropdown_menu():
        new_menu = []
        calib_file_ext = '.palm_etof'
        if os.path.isdir(path_textinput.value):
            with os.scandir(path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith((calib_file_ext)):
                        new_menu.append((entry.name[:-len(calib_file_ext)], entry.name))
            load_dropdown.button_type = 'default'
            load_dropdown.menu = sorted(new_menu, reverse=True)
        else:
            load_dropdown.button_type = 'danger'
            load_dropdown.menu = new_menu

    doc.add_next_tick_callback(update_load_dropdown_menu)
    doc.add_periodic_callback(update_load_dropdown_menu, 5000)

    load_dropdown = Dropdown(label="Load", menu=[], width=135)
    load_dropdown.on_click(load_dropdown_callback)


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
        column(waveform_plot, fit_plot, thz_scan_plot), Spacer(width=30),
        column(
            path_textinput, calibrate_button, row(save_button, Spacer(width=10), load_dropdown),
            datatable, fit_eq_div, calib_const_div))

    return Panel(child=tab_calibration_layout, title="Calibration")
