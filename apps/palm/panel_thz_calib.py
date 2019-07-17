import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    Button,
    Circle,
    ColumnDataSource,
    DataRange1d,
    Div,
    Dropdown,
    Grid,
    Line,
    LinearAxis,
    Panel,
    PanTool,
    Plot,
    ResetTool,
    Spacer,
    TextInput,
    Title,
    WheelZoomTool,
)

PLOT_CANVAS_WIDTH = 620
PLOT_CANVAS_HEIGHT = 380


def create(palm):
    fit_max = 1
    fit_min = 0

    doc = curdoc()

    # THz calibration plot
    scan_plot = Plot(
        title=Title(text="THz calibration"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
    )

    # ---- tools
    scan_plot.toolbar.logo = None
    scan_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    scan_plot.add_layout(LinearAxis(axis_label='Stage delay motor'), place='below')
    scan_plot.add_layout(
        LinearAxis(axis_label='Energy shift, eV', major_label_orientation='vertical'), place='left'
    )

    # ---- grid lines
    scan_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    scan_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- circle cluster glyphs
    scan_circle_source = ColumnDataSource(dict(x=[], y=[]))
    scan_plot.add_glyph(scan_circle_source, Circle(x='x', y='y', line_alpha=0, fill_alpha=0.5))

    # ---- circle glyphs
    scan_avg_circle_source = ColumnDataSource(dict(x=[], y=[]))
    scan_plot.add_glyph(
        scan_avg_circle_source, Circle(x='x', y='y', line_color='purple', fill_color='purple')
    )

    # ---- line glyphs
    fit_line_source = ColumnDataSource(dict(x=[], y=[]))
    scan_plot.add_glyph(fit_line_source, Line(x='x', y='y', line_color='purple'))

    # THz calibration folder path text input
    def path_textinput_callback(_attr, _old, _new):
        update_load_dropdown_menu()
        path_periodic_update()

    path_textinput = TextInput(
        title="THz calibration path:", value=os.path.join(os.path.expanduser('~')), width=525
    )
    path_textinput.on_change('value', path_textinput_callback)

    # THz calibration eco scans dropdown
    def scans_dropdown_callback(_attr, _old, new):
        scans_dropdown.label = new

    scans_dropdown = Dropdown(label="ECO scans", button_type='default', menu=[])
    scans_dropdown.on_change('value', scans_dropdown_callback)

    # ---- eco scans periodic update
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
        palm.calibrate_thz(path=os.path.join(path_textinput.value, scans_dropdown.value))
        fit_max_textinput.value = str(np.ceil(palm.thz_calib_data.index.values.max()))
        fit_min_textinput.value = str(np.floor(palm.thz_calib_data.index.values.min()))
        update_calibration_plot()

    def update_calibration_plot():
        scan_plot.xaxis.axis_label = '{}, {}'.format(palm.thz_motor_name, palm.thz_motor_unit)

        scan_circle_source.data.update(
            x=np.repeat(
                palm.thz_calib_data.index, palm.thz_calib_data['peak_shift'].apply(len)
            ).tolist(),
            y=np.concatenate(palm.thz_calib_data['peak_shift'].values).tolist(),
        )

        scan_avg_circle_source.data.update(
            x=palm.thz_calib_data.index.tolist(), y=palm.thz_calib_data['peak_shift_mean'].tolist()
        )

        x = np.linspace(fit_min, fit_max, 100)
        y = palm.thz_slope * x + palm.thz_intersect
        fit_line_source.data.update(x=np.round(x, decimals=5), y=np.round(y, decimals=5))

        calib_const_div.text = """
        thz_slope = {}
        """.format(
            palm.thz_slope
        )

    calibrate_button = Button(label="Calibrate THz", button_type='default')
    calibrate_button.on_click(calibrate_button_callback)

    # THz fit maximal value text input
    def fit_max_textinput_callback(_attr, old, new):
        nonlocal fit_max
        try:
            new_value = float(new)
            if new_value > fit_min:
                fit_max = new_value
                palm.calibrate_thz(
                    path=os.path.join(path_textinput.value, scans_dropdown.value),
                    fit_range=(fit_min, fit_max),
                )
                update_calibration_plot()
            else:
                fit_max_textinput.value = old

        except ValueError:
            fit_max_textinput.value = old

    fit_max_textinput = TextInput(title='Maximal fit value:', value=str(fit_max))
    fit_max_textinput.on_change('value', fit_max_textinput_callback)

    # THz fit maximal value text input
    def fit_min_textinput_callback(_attr, old, new):
        nonlocal fit_min
        try:
            new_value = float(new)
            if new_value < fit_max:
                fit_min = new_value
                palm.calibrate_thz(
                    path=os.path.join(path_textinput.value, scans_dropdown.value),
                    fit_range=(fit_min, fit_max),
                )
                update_calibration_plot()
            else:
                fit_min_textinput.value = old

        except ValueError:
            fit_min_textinput.value = old

    fit_min_textinput = TextInput(title='Minimal fit value:', value=str(fit_min))
    fit_min_textinput.on_change('value', fit_min_textinput_callback)

    # Save calibration button
    def save_button_callback():
        palm.save_thz_calib(path=path_textinput.value)
        update_load_dropdown_menu()

    save_button = Button(label="Save", button_type='default', width=135)
    save_button.on_click(save_button_callback)

    # Load calibration button
    def load_dropdown_callback(_attr, _old, new):
        palm.load_thz_calib(os.path.join(path_textinput.value, new))
        update_calibration_plot()

    def update_load_dropdown_menu():
        new_menu = []
        calib_file_ext = '.palm_thz'
        if os.path.isdir(path_textinput.value):
            for entry in os.scandir(path_textinput.value):
                if entry.is_file() and entry.name.endswith((calib_file_ext)):
                    new_menu.append((entry.name[: -len(calib_file_ext)], entry.name))
            load_dropdown.button_type = 'default'
            load_dropdown.menu = sorted(new_menu, reverse=True)
        else:
            load_dropdown.button_type = 'danger'
            load_dropdown.menu = new_menu

    doc.add_next_tick_callback(update_load_dropdown_menu)
    doc.add_periodic_callback(update_load_dropdown_menu, 5000)

    load_dropdown = Dropdown(label="Load", menu=[], width=135)
    load_dropdown.on_change('value', load_dropdown_callback)

    # Calibration constants
    calib_const_div = Div(
        text="""
        thz_slope = {}
        """.format(
            0
        )
    )

    # assemble
    tab_layout = column(
        row(
            scan_plot,
            Spacer(width=30),
            column(
                path_textinput,
                scans_dropdown,
                calibrate_button,
                fit_max_textinput,
                fit_min_textinput,
                row(save_button, Spacer(width=10), load_dropdown),
                calib_const_div,
            ),
        )
    )

    return Panel(child=tab_layout, title="THz Calibration")
