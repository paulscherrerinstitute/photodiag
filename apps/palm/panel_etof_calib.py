import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, CheckboxEditor, \
    Circle, ColumnDataSource, DataRange1d, DataTable, Div, Dropdown, Grid, \
    HoverTool, IntEditor, Legend, Line, LinearAxis, MultiLine, Panel, PanTool, \
    Plot, ResetTool, Spacer, Span, TableColumn, TextInput, Title, WheelZoomTool

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
    waveform_plot_hovertool = HoverTool(
        tooltips=[
            ("energy, eV", '@en'),
            ("eTOF bin", '$x{0.}'),
            ])

    waveform_plot.add_tools(
        PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), waveform_plot_hovertool)

    # ---- axes
    waveform_plot.add_layout(
        LinearAxis(axis_label='eTOF time bin'), place='below')
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


    # Calibration results datatables
    def datatable_ref_source_callback(_attr, _old, new):
        for en, ps, use in zip(new['energy'], new['peak_pos_ref'], new['use_in_fit']):
            palm.etofs['0'].calib_data.loc[en, 'calib_tpeak'] = (ps if ps != 'NaN' else np.nan)
            palm.etofs['0'].calib_data.loc[en, 'use_in_fit'] = use

        calib_res = {}
        for etof_key in palm.etofs:
            calib_res[etof_key] = palm.etofs[etof_key].fit_calibration_curve()
        update_calibration_plot(calib_res)

    datatable_ref_source = ColumnDataSource(
        dict(energy=['', '', ''], peak_pos_ref=['', '', ''], use_in_fit=[True, True, True]))
    datatable_ref_source.on_change('data', datatable_ref_source_callback)

    datatable_ref = DataTable(
        source=datatable_ref_source,
        columns=[
            TableColumn(field='energy', title="Photon Energy, eV", editor=IntEditor()),
            TableColumn(field='peak_pos_ref', title="Reference Peak", editor=IntEditor()),
            TableColumn(field='use_in_fit', title=" ", editor=CheckboxEditor(), width=80)],
        index_position=None,
        editable=True,
        height=300,
        width=250,
    )

    def datatable_str_source_callback(_attr, _old, new):
        for en, ps, use in zip(new['energy'], new['peak_pos_str'], new['use_in_fit']):
            palm.etofs['1'].calib_data.loc[en, 'calib_tpeak'] = (ps if ps != 'NaN' else np.nan)
            palm.etofs['1'].calib_data.loc[en, 'use_in_fit'] = use

        calib_res = {}
        for etof_key in palm.etofs:
            calib_res[etof_key] = palm.etofs[etof_key].fit_calibration_curve()
        update_calibration_plot(calib_res)

    datatable_str_source = ColumnDataSource(
        dict(energy=['', '', ''], peak_pos_str=['', '', ''], use_in_fit=[True, True, True]))
    datatable_str_source.on_change('data', datatable_str_source_callback)

    datatable_str = DataTable(
        source=datatable_str_source,
        columns=[
            TableColumn(field='energy', title="Photon Energy, eV", editor=IntEditor()),
            TableColumn(field='peak_pos_str', title="Streaked Peak", editor=IntEditor()),
            TableColumn(field='use_in_fit', title=" ", editor=CheckboxEditor(), width=80)],
        index_position=None,
        editable=True,
        height=350,
        width=250,
    )


    # eTOF calibration folder path text input
    def path_textinput_callback(_attr, _old, _new):
        path_periodic_update()
        update_load_dropdown_menu()

    path_textinput = TextInput(
        title="eTOF calibration path:", value=os.path.join(os.path.expanduser('~')), width=525)
    path_textinput.on_change('value', path_textinput_callback)


    # eTOF calibration eco scans dropdown
    def scans_dropdown_callback(selection):
        scans_dropdown.label = selection

    scans_dropdown = Dropdown(label="ECO scans", button_type='default', menu=[])
    scans_dropdown.on_click(scans_dropdown_callback)

    # ---- etof scans periodic update
    def path_periodic_update():
        new_menu = []
        if os.path.isdir(path_textinput.value):
            for entry in os.scandir(path_textinput.value):
                if entry.is_file() and entry.name.endswith('.json'):
                    new_menu.append((entry.name, entry.name))
        scans_dropdown.menu = sorted(new_menu, reverse=True)

    doc.add_periodic_callback(path_periodic_update, 5000)


    # Calibrate button
    def calibrate_button_callback():
        try:
            palm.calibrate_etof_eco(
                eco_scan_filename=os.path.join(path_textinput.value, scans_dropdown.value))
        except Exception:
            palm.calibrate_etof(folder_name=path_textinput.value)

        datatable_ref_source.data.update(
            energy=palm.etofs['0'].calib_data.index.tolist(),
            peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
            use_in_fit=palm.etofs['0'].calib_data['use_in_fit'].tolist())

        datatable_str_source.data.update(
            energy=palm.etofs['0'].calib_data.index.tolist(),
            peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist(),
            use_in_fit=palm.etofs['1'].calib_data['use_in_fit'].tolist())

    def update_calibration_plot(calib_res):
        etof_ref = palm.etofs['0']
        etof_str = palm.etofs['1']

        shift_val = 0
        etof_ref_wf_shifted = []
        etof_str_wf_shifted = []
        for wf_ref, wf_str in zip(etof_ref.calib_data['waveform'], etof_str.calib_data['waveform']):
            shift_val -= max(wf_ref.max(), wf_str.max())
            etof_ref_wf_shifted.append(wf_ref + shift_val)
            etof_str_wf_shifted.append(wf_str + shift_val)

        waveform_ref_source.data.update(
            xs=len(etof_ref.calib_data)*[list(range(etof_ref.internal_time_bins))],
            ys=etof_ref_wf_shifted,
            en=etof_ref.calib_data.index.tolist())

        waveform_str_source.data.update(
            xs=len(etof_str.calib_data)*[list(range(etof_str.internal_time_bins))],
            ys=etof_str_wf_shifted,
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

    calibrate_button = Button(label="Calibrate eTOF", button_type='default')
    calibrate_button.on_click(calibrate_button_callback)


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
    def save_button_callback():
        palm.save_etof_calib(path=path_textinput.value)
        update_load_dropdown_menu()

    save_button = Button(label="Save", button_type='default', width=135)
    save_button.on_click(save_button_callback)


    # Load calibration button
    def load_dropdown_callback(selection):
        if selection:
            palm.load_etof_calib(os.path.join(path_textinput.value, selection))

            datatable_ref_source.data.update(
                energy=palm.etofs['0'].calib_data.index.tolist(),
                peak_pos_ref=palm.etofs['0'].calib_data['calib_tpeak'].tolist(),
                use_in_fit=palm.etofs['0'].calib_data['use_in_fit'].tolist())

            datatable_str_source.data.update(
                energy=palm.etofs['0'].calib_data.index.tolist(),
                peak_pos_str=palm.etofs['1'].calib_data['calib_tpeak'].tolist(),
                use_in_fit=palm.etofs['1'].calib_data['use_in_fit'].tolist())

            # Drop selection, so that this callback can be triggered again on the same dropdown menu
            # item from the user perspective
            load_dropdown.value = ''

    def update_load_dropdown_menu():
        new_menu = []
        calib_file_ext = '.palm_etof'
        if os.path.isdir(path_textinput.value):
            for entry in os.scandir(path_textinput.value):
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


    # eTOF fitting equation
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
    tab_layout = column(
        row(
            column(waveform_plot, fit_plot), Spacer(width=30),
            column(
                path_textinput, scans_dropdown, calibrate_button,
                phot_peak_noise_thr_textinput, el_peak_noise_thr_textinput,
                row(save_button, Spacer(width=10), load_dropdown),
                row(datatable_ref, Spacer(width=10), datatable_str),
                fit_eq_div, calib_const_div)))

    return Panel(child=tab_layout, title="eTOF Calibration")
