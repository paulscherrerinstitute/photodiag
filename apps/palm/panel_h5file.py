import os
from functools import partial

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, CheckboxButtonGroup, \
    ColumnDataSource, DataRange1d, Dropdown, Grid, Legend, Line, LinearAxis, Panel, \
    PanTool, Plot, ResetTool, Slider, Spacer, Span, TextInput, Title, WheelZoomTool

PLOT_CANVAS_WIDTH = 620
PLOT_CANVAS_HEIGHT = 380

def create(palm):
    energy_min = palm.energy_range.min()
    energy_max = palm.energy_range.max()
    energy_npoints = palm.energy_range.size

    current_results = (0, 0, 0, 0)

    doc = curdoc()

    # Streaked and reference waveforms plot
    waveform_plot = Plot(
        title=Title(text="eTOF waveforms"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    waveform_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

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
    waveform_ref_line = waveform_plot.add_glyph(
        waveform_source, Line(x='x_ref', y='y_ref', line_color='blue'))
    waveform_str_line = waveform_plot.add_glyph(
        waveform_source, Line(x='x_str', y='y_str', line_color='red'))

    # ---- legend
    waveform_plot.add_layout(Legend(items=[
        ("reference", [waveform_ref_line]),
        ("streaked", [waveform_str_line])
    ]))
    waveform_plot.legend.click_policy = "hide"


    # Cross-correlation plot
    xcorr_plot = Plot(
        title=Title(text="Waveforms cross-correlation"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    xcorr_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    xcorr_plot.add_layout(
        LinearAxis(axis_label='Energy shift, eV'), place='below')
    xcorr_plot.add_layout(
        LinearAxis(axis_label='Cross-correlation', major_label_orientation='vertical'),
        place='left')

    # ---- grid lines
    xcorr_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    xcorr_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- line glyphs
    xcorr_source = ColumnDataSource(dict(lags=[], xcorr1=[], xcorr2=[]))
    xcorr_plot.add_glyph(
        xcorr_source, Line(x='lags', y='xcorr1', line_color='purple', line_dash='dashed'))
    xcorr_plot.add_glyph(xcorr_source, Line(x='lags', y='xcorr2', line_color='purple'))

    # ---- vertical span
    xcorr_center_span = Span(location=0, dimension='height')
    xcorr_plot.add_layout(xcorr_center_span)


    # Delays plot
    pulse_delay_plot = Plot(
        title=Title(text="Pulse delays"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    pulse_delay_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    pulse_delay_plot.add_layout(
        LinearAxis(axis_label='Pulse number'), place='below')
    pulse_delay_plot.add_layout(
        LinearAxis(axis_label='Pulse delay (uncalib), eV', major_label_orientation='vertical'),
        place='left')

    # ---- grid lines
    pulse_delay_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    pulse_delay_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- line glyphs
    pulse_delay_source = ColumnDataSource(dict(pulse=[], delay=[]))
    pulse_delay_plot.add_glyph(
        pulse_delay_source, Line(x='pulse', y='delay', line_color='steelblue'))

    # ---- vertical span
    pulse_delay_plot_span = Span(location=0, dimension='height')
    pulse_delay_plot.add_layout(pulse_delay_plot_span)


    # Pulse lengths plot
    pulse_length_plot = Plot(
        title=Title(text="Pulse lengths"),
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=PLOT_CANVAS_HEIGHT,
        plot_width=PLOT_CANVAS_WIDTH,
        toolbar_location='right',
        logo=None,
    )

    # ---- tools
    pulse_length_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool())

    # ---- axes
    pulse_length_plot.add_layout(
        LinearAxis(axis_label='Pulse number'), place='below')
    pulse_length_plot.add_layout(
        LinearAxis(axis_label='Pulse length (uncalib), eV', major_label_orientation='vertical'),
        place='left')

    # ---- grid lines
    pulse_length_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    pulse_length_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    # ---- line glyphs
    pulse_length_source = ColumnDataSource(dict(x=[], y=[]))
    pulse_length_plot.add_glyph(pulse_length_source, Line(x='x', y='y', line_color='steelblue'))

    # ---- vertical span
    pulse_length_plot_span = Span(location=0, dimension='height')
    pulse_length_plot.add_layout(pulse_length_plot_span)


    # Folder path text input
    def path_textinput_callback(_attr, _old, new):
        save_textinput.value = new
        path_periodic_update()

    path_textinput = TextInput(
        title="Folder Path:", value=os.path.join(os.path.expanduser('~')), width=525)
    path_textinput.on_change('value', path_textinput_callback)


    # Saved runs dropdown menu
    def h5_update(pulse, delays, debug_data):
        prep_data, lags, corr_res_uncut, corr_results = debug_data

        waveform_source.data.update(
            x_str=palm.energy_range, y_str=prep_data['1'][pulse, :],
            x_ref=palm.energy_range, y_ref=prep_data['0'][pulse, :])

        xcorr_source.data.update(
            lags=lags, xcorr1=corr_res_uncut[pulse, :], xcorr2=corr_results[pulse, :])

        xcorr_center_span.location = delays[pulse]
        pulse_delay_plot_span.location = pulse
        pulse_length_plot_span.location = pulse

    # this placeholder function should be reassigned in 'saved_runs_dropdown_callback'
    h5_update_fun = lambda pulse: None

    def saved_runs_dropdown_callback(selection):
        if selection != "Saved Runs":
            nonlocal h5_update_fun, current_results
            saved_runs_dropdown.label = selection
            filepath = os.path.join(path_textinput.value, selection)
            tags, delays, lengths, debug_data = palm.process_hdf5_file(filepath, debug=True)
            current_results = (selection, tags, delays, lengths)

            if autosave_checkbox.active:
                save_button_callback()

            pulse_delay_source.data.update(pulse=np.arange(len(delays)), delay=delays)
            pulse_length_source.data.update(x=np.arange(len(lengths)), y=lengths)
            h5_update_fun = partial(h5_update, delays=delays, debug_data=debug_data)

            pulse_slider.end = len(delays) - 1
            pulse_slider.value = 0
            h5_update_fun(0)

    saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[])
    saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

    # ---- saved run periodic update
    def path_periodic_update():
        new_menu = []
        if os.path.isdir(path_textinput.value):
            with os.scandir(path_textinput.value) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                        new_menu.append((entry.name, entry.name))
        saved_runs_dropdown.menu = sorted(new_menu, reverse=True)

    doc.add_periodic_callback(path_periodic_update, 10000)


    # Pulse number slider
    def pulse_slider_callback(_attr, _old, new):
        h5_update_fun(pulse=new)

    pulse_slider = Slider(start=0, end=99999, value=0, step=1, title="Pulse ID", width=500)
    pulse_slider.on_change('value', pulse_slider_callback)


    # Energy maximal range value text input
    def energy_max_textinput_callback(_attr, old, new):
        nonlocal energy_max
        try:
            new_value = float(new)
            if new_value > energy_min:
                energy_max = new_value
                palm.energy_range = np.linspace(energy_min, energy_max, energy_npoints)
                saved_runs_dropdown_callback(saved_runs_dropdown.label)
            else:
                energy_max_textinput.value = old

        except ValueError:
            energy_max_textinput.value = old

    energy_max_textinput = TextInput(title='Maximal Energy, eV:', value=str(energy_max))
    energy_max_textinput.on_change('value', energy_max_textinput_callback)


    # Energy minimal range value text input
    def energy_min_textinput_callback(_attr, old, new):
        nonlocal energy_min
        try:
            new_value = float(new)
            if new_value < energy_max:
                energy_min = new_value
                palm.energy_range = np.linspace(energy_min, energy_max, energy_npoints)
                saved_runs_dropdown_callback(saved_runs_dropdown.label)
            else:
                energy_min_textinput.value = old

        except ValueError:
            energy_min_textinput.value = old

    energy_min_textinput = TextInput(title='Minimal Energy, eV:', value=str(energy_min))
    energy_min_textinput.on_change('value', energy_min_textinput_callback)


    # Energy number of interpolation points text input
    def energy_npoints_textinput_callback(_attr, old, new):
        nonlocal energy_npoints
        try:
            new_value = int(new)
            if new_value > 1:
                energy_npoints = new_value
                palm.energy_range = np.linspace(energy_min, energy_max, energy_npoints)
                saved_runs_dropdown_callback(saved_runs_dropdown.label)
            else:
                energy_npoints_textinput.value = old

        except ValueError:
            energy_npoints_textinput.value = old

    energy_npoints_textinput = TextInput(
        title='Number of interpolation points:', value=str(energy_npoints))
    energy_npoints_textinput.on_change('value', energy_npoints_textinput_callback)


    # Save location
    save_textinput = TextInput(
        title="Save Folder Path:", value=os.path.join(os.path.expanduser('~')), width=525)


    # Autosave checkbox
    autosave_checkbox = CheckboxButtonGroup(labels=["Auto Save"], active=[], width=100)


    # Save button
    def save_button_callback():
        if current_results[0]:
            filename, tags, delays, lengths = current_results
            save_filename = os.path.splitext(filename)[0] + '.csv'
            df = pd.DataFrame({'pulse_id': tags, 'pulse_delay': delays, 'pulse_length': lengths})
            df.to_csv(os.path.join(save_textinput.value, save_filename), index=False)

    save_button = Button(label="Save Results", button_type='default')
    save_button.on_click(save_button_callback)


    # assemble
    tab_h5file_layout = column(
        row(
            column(waveform_plot, xcorr_plot), Spacer(width=30),
            column(
                path_textinput, saved_runs_dropdown, pulse_slider,
                Spacer(height=30),
                energy_min_textinput, energy_max_textinput, energy_npoints_textinput,
                Spacer(height=30),
                save_textinput, autosave_checkbox, save_button)),
        row(pulse_delay_plot, Spacer(width=10), pulse_length_plot))

    return Panel(child=tab_h5file_layout, title="HDF5 File")
