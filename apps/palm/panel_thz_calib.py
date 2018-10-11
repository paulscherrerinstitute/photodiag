import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, Circle, \
    ColumnDataSource, DataRange1d, Div, Dropdown, Grid, Line, LinearAxis, \
    Panel, PanTool, Plot, ResetTool, Spacer, TextInput, Title, WheelZoomTool

PLOT_CANVAS_WIDTH = 620
PLOT_CANVAS_HEIGHT = 380

def create(palm):
    thz_fit_max = 1
    thz_fit_min = 0

    doc = curdoc()

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
        LinearAxis(axis_label='Stage delay motor'), place='below')
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


    # THz calibration folder path text input
    def thz_path_textinput_callback(_attr, _old, _new):
        update_thz_load_dropdown_menu()
        path_periodic_update()

    thz_path_textinput = TextInput(
        title="THz calibration path:", value=os.path.join(os.path.expanduser('~')), width=525)
    thz_path_textinput.on_change('value', thz_path_textinput_callback)


    # THz calibration eco scans dropdown
    def thz_scans_dropdown_callback(selection):
        thz_scans_dropdown.label = selection

    thz_scans_dropdown = Dropdown(label="ECO scans", button_type='default', menu=[])
    thz_scans_dropdown.on_click(thz_scans_dropdown_callback)

    # ---- eco scans periodic update
    def path_periodic_update():
        new_menu = []
        if os.path.isdir(thz_path_textinput.value):
            with os.scandir(thz_path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith('.json'):
                        new_menu.append((entry.name, entry.name))
        thz_scans_dropdown.menu = sorted(new_menu, reverse=True)

    doc.add_periodic_callback(path_periodic_update, 5000)


    # Calibrate button
    def thz_calibrate_button_callback():
        palm.calibrate_thz(path=os.path.join(thz_path_textinput.value, thz_scans_dropdown.value))
        thz_fit_max_textinput.value = str(np.ceil(palm.thz_calib_data.index.values.max()))
        thz_fit_min_textinput.value = str(np.floor(palm.thz_calib_data.index.values.min()))
        update_thz_calibration_plot()

    def update_thz_calibration_plot():
        thz_scan_plot.xaxis.axis_label = f'{palm.thz_motor_name}, {palm.thz_motor_unit}'

        thz_scan_circle_source.data.update(
            x=palm.thz_calib_data.index.tolist(),
            y=palm.thz_calib_data['peak_shift_mean'].tolist())

        x = np.linspace(thz_fit_min, thz_fit_max, 100)
        y = palm.thz_slope * x + palm.thz_intersect
        thz_fit_line_source.data.update(x=x, y=y)

        thz_calib_const_div.text = f"""
        thz_slope = {palm.thz_slope}
        """

    thz_calibrate_button = Button(label="Calibrate THz", button_type='default')
    thz_calibrate_button.on_click(thz_calibrate_button_callback)


    # THz fit maximal value text input
    def thz_fit_max_textinput_callback(_attr, old, new):
        nonlocal thz_fit_max
        try:
            new_value = float(new)
            if new_value > thz_fit_min:
                thz_fit_max = new_value
                palm.calibrate_thz(
                    path=os.path.join(thz_path_textinput.value, thz_scans_dropdown.value),
                    fit_range=(thz_fit_min, thz_fit_max))
                update_thz_calibration_plot()
            else:
                thz_fit_max_textinput.value = old

        except ValueError:
            thz_fit_max_textinput.value = old

    thz_fit_max_textinput = TextInput(title='Maximal fit value:', value=str(thz_fit_max))
    thz_fit_max_textinput.on_change('value', thz_fit_max_textinput_callback)


    # THz fit maximal value text input
    def thz_fit_min_textinput_callback(_attr, old, new):
        nonlocal thz_fit_min
        try:
            new_value = float(new)
            if new_value < thz_fit_max:
                thz_fit_min = new_value
                palm.calibrate_thz(
                    path=os.path.join(thz_path_textinput.value, thz_scans_dropdown.value),
                    fit_range=(thz_fit_min, thz_fit_max))
                update_thz_calibration_plot()
            else:
                thz_fit_min_textinput.value = old

        except ValueError:
            thz_fit_min_textinput.value = old

    thz_fit_min_textinput = TextInput(title='Minimal fit value:', value=str(thz_fit_min))
    thz_fit_min_textinput.on_change('value', thz_fit_min_textinput_callback)


    # Save calibration button
    def thz_save_button_callback():
        palm.save_thz_calib(path=thz_path_textinput.value)
        update_thz_load_dropdown_menu()

    thz_save_button = Button(label="Save", button_type='default', width=135)
    thz_save_button.on_click(thz_save_button_callback)


    # Load calibration button
    def thz_load_dropdown_callback(selection):
        palm.load_thz_calib(os.path.join(thz_path_textinput.value, selection))
        update_thz_calibration_plot()

    def update_thz_load_dropdown_menu():
        new_menu = []
        calib_file_ext = '.palm_thz'
        if os.path.isdir(thz_path_textinput.value):
            with os.scandir(thz_path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith((calib_file_ext)):
                        new_menu.append((entry.name[:-len(calib_file_ext)], entry.name))
            thz_load_dropdown.button_type = 'default'
            thz_load_dropdown.menu = sorted(new_menu, reverse=True)
        else:
            thz_load_dropdown.button_type = 'danger'
            thz_load_dropdown.menu = new_menu

    doc.add_next_tick_callback(update_thz_load_dropdown_menu)
    doc.add_periodic_callback(update_thz_load_dropdown_menu, 5000)

    thz_load_dropdown = Dropdown(label="Load", menu=[], width=135)
    thz_load_dropdown.on_click(thz_load_dropdown_callback)


    # Calibration constants
    thz_calib_const_div = Div(
        text=f"""
        thz_slope = {0}
        """)


    # assemble
    tab_calibration_layout = column(
        row(
            thz_scan_plot, Spacer(width=30),
            column(
                thz_path_textinput, thz_scans_dropdown, thz_calibrate_button,
                thz_fit_max_textinput, thz_fit_min_textinput,
                row(thz_save_button, Spacer(width=10), thz_load_dropdown),
                thz_calib_const_div)))

    return Panel(child=tab_calibration_layout, title="THz Calibration")
