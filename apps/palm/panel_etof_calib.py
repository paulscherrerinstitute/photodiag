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
        plot_height=760,
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
        update_etof_calibration_plot(calib_res)

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
        height=400,
        width=500,
    )


    # eTOF calibration folder path text input
    def etof_path_textinput_callback(_attr, _old, _new):
        etof_path_periodic_update()
        update_etof_load_dropdown_menu()

    etof_path_textinput = TextInput(
        title="eTOF calibration path:", value=os.path.join(os.path.expanduser('~')), width=525)
    etof_path_textinput.on_change('value', etof_path_textinput_callback)


    # eTOF calibration eco scans dropdown
    def etof_scans_dropdown_callback(selection):
        etof_scans_dropdown.label = selection

    etof_scans_dropdown = Dropdown(label="ECO scans", button_type='default', menu=[])
    etof_scans_dropdown.on_click(etof_scans_dropdown_callback)

    # ---- etof scans periodic update
    def etof_path_periodic_update():
        new_menu = []
        if os.path.isdir(etof_path_textinput.value):
            with os.scandir(etof_path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith('.json'):
                        new_menu.append((entry.name, entry.name))
        etof_scans_dropdown.menu = sorted(new_menu, reverse=True)

    doc.add_periodic_callback(etof_path_periodic_update, 5000)


    # Calibrate button
    def etof_calibrate_button_callback():
        palm.calibrate_etof_eco(
            eco_scan_filename=os.path.join(etof_path_textinput.value, etof_scans_dropdown.value))

        datatable_source.data.update(
            energy=palm.etofs['0'].calib_data.index.tolist(),
            peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
            peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

    def update_etof_calibration_plot(calib_res):
        etof_ref = palm.etofs['0']
        etof_str = palm.etofs['1']

        etof_ref_wf_shift = []
        etof_str_wf_shift = []
        shift_val = 0
        for wf_ref, wf_str in zip(etof_ref.calib_data['waveform'], etof_str.calib_data['waveform']):
            etof_ref_wf_shift.append(wf_ref + shift_val)
            etof_str_wf_shift.append(wf_str + shift_val)
            shift_val += max(wf_ref.max(), wf_str.max())

        waveform_ref_source.data.update(
            xs=len(etof_ref.calib_data)*[list(range(etof_ref.internal_time_bins))],
            ys=etof_ref_wf_shift,
            en=etof_ref.calib_data.index.tolist())

        waveform_str_source.data.update(
            xs=len(etof_str.calib_data)*[list(range(etof_str.internal_time_bins))],
            ys=etof_str_wf_shift,
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

        etof_calib_const_div.text = f"""
        a_str = {etof_str.calib_a:.2f}<br>
        b_str = {etof_str.calib_b:.2f}<br>
        <br>
        a_ref = {etof_ref.calib_a:.2f}<br>
        b_ref = {etof_ref.calib_b:.2f}
        """

    etof_calibrate_button = Button(label="Calibrate eTOF", button_type='default')
    etof_calibrate_button.on_click(etof_calibrate_button_callback)


    # Photon peak noise threshold value text input
    def phot_peak_noise_thr_textinput_callback(_attr, old, new):
        try:
            new_value = float(new)
            if new_value > 0:
                for etof in palm.etofs.values():
                    etof.photon_peak_noise_thr = new_value
            else:
                phot_peak_noise_thr_textinput.value = old

        except ValueError:
            phot_peak_noise_thr_textinput.value = old

    phot_peak_noise_thr_textinput = TextInput(title='Photon peak noise threshold:', value=str(1))
    phot_peak_noise_thr_textinput.on_change('value', phot_peak_noise_thr_textinput_callback)


    # Electron peak noise threshold value text input
    def el_peak_noise_thr_textinput_callback(_attr, old, new):
        try:
            new_value = float(new)
            if new_value > 0:
                for etof in palm.etofs.values():
                    etof.electron_peak_noise_thr = new_value
            else:
                el_peak_noise_thr_textinput.value = old

        except ValueError:
            el_peak_noise_thr_textinput.value = old

    el_peak_noise_thr_textinput = TextInput(title='Electron peak noise threshold:', value=str(10))
    el_peak_noise_thr_textinput.on_change('value', el_peak_noise_thr_textinput_callback)


    # Save calibration button
    def etof_save_button_callback():
        palm.save_etof_calib(path=etof_path_textinput.value)
        update_etof_load_dropdown_menu()

    etof_save_button = Button(label="Save", button_type='default', width=135)
    etof_save_button.on_click(etof_save_button_callback)


    # Load calibration button
    def etof_load_dropdown_callback(selection):
        if selection:
            palm.load_etof_calib(os.path.join(etof_path_textinput.value, selection))

            datatable_source.data.update(
                energy=palm.etofs['0'].calib_data.index.tolist(),
                peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
                peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist())

            # Drop selection, so that this callback can be triggered again on the same dropdown menu
            # item from the user perspective
            etof_load_dropdown.value = ''

    def update_etof_load_dropdown_menu():
        new_menu = []
        calib_file_ext = '.palm_etof'
        if os.path.isdir(etof_path_textinput.value):
            with os.scandir(etof_path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith((calib_file_ext)):
                        new_menu.append((entry.name[:-len(calib_file_ext)], entry.name))
            etof_load_dropdown.button_type = 'default'
            etof_load_dropdown.menu = sorted(new_menu, reverse=True)
        else:
            etof_load_dropdown.button_type = 'danger'
            etof_load_dropdown.menu = new_menu

    doc.add_next_tick_callback(update_etof_load_dropdown_menu)
    doc.add_periodic_callback(update_etof_load_dropdown_menu, 5000)

    etof_load_dropdown = Dropdown(label="Load", menu=[], width=135)
    etof_load_dropdown.on_click(etof_load_dropdown_callback)


    # eTOF fitting equation
    etof_fit_eq_div = Div(text="""Fitting equation:<br><br><img src="/palm/static/5euwuy.gif">""")


    # Calibration constants
    etof_calib_const_div = Div(
        text=f"""
        a_str = {0}<br>
        b_str = {0}<br>
        <br>
        a_ref = {0}<br>
        b_ref = {0}
        """)


    # assemble
    tab_calibration_layout = column(
        row(
            column(waveform_plot, fit_plot), Spacer(width=30),
            column(
                etof_path_textinput, etof_scans_dropdown, etof_calibrate_button,
                phot_peak_noise_thr_textinput, el_peak_noise_thr_textinput,
                row(etof_save_button, Spacer(width=10), etof_load_dropdown),
                datatable, etof_fit_eq_div, etof_calib_const_div)))

    return Panel(child=tab_calibration_layout, title="eTOF Calibration")
