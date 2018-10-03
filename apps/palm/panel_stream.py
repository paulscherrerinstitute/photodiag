from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BasicTicker, Button, ColumnDataSource, CustomJS, \
    DataRange1d, Grid, Legend, Line, LinearAxis, Panel, PanTool, Plot, \
    ResetTool, Slider, Spacer, Span, Title, Toggle, WheelZoomTool
from tornado import gen

import receiver

PLOT_CANVAS_WIDTH = 620
PLOT_CANVAS_HEIGHT = 380

def create(palm):
    connected = False
    current_message = None
    stream_t = 0

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
    waveform_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

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
    xcorr_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

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
    pulse_delay_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

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
    pulse_length_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())

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


    # Image buffer slider
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


    # Connect toggle button
    def connect_toggle_callback(state):
        nonlocal connected
        if state:
            connected = True
            connect_toggle.label = 'Connecting'
            connect_toggle.button_type = 'default'

        else:
            connected = False
            connect_toggle.label = 'Connect'
            connect_toggle.button_type = 'default'

    connect_toggle = Toggle(label="Connect", button_type='default')
    connect_toggle.on_click(connect_toggle_callback)


    # Intensity stream reset button
    def reset_button_callback():
        nonlocal stream_t
        stream_t = 1  # keep the latest point in order to prevent full axis reset

    reset_button = Button(label="Reset", button_type='default')
    reset_button.on_click(reset_button_callback)


    # Stream update coroutine
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


    # Periodic callback to fetch data from receiver
    @gen.coroutine
    def internal_periodic_callback():
        nonlocal current_message
        if waveform_plot.inner_width is None:
            # wait for the initialization to finish, thus skip this periodic callback
            return

        if connected:
            if receiver.state == 'polling':
                connect_toggle.label = 'Polling'
                connect_toggle.button_type = 'warning'

            elif receiver.state == 'receiving':
                connect_toggle.label = 'Receiving'
                connect_toggle.button_type = 'success'

                # Set slider to the right-most position
                if len(receiver.data_buffer) > 1:
                    buffer_slider.end = len(receiver.data_buffer) - 1
                    buffer_slider.value = len(receiver.data_buffer) - 1

                if receiver.data_buffer:
                    current_message = receiver.data_buffer[-1]

        doc.add_next_tick_callback(partial(update, message=current_message))

    doc.add_periodic_callback(internal_periodic_callback, 1000)


    # assemble
    tab_stream_layout = column(
        row(
            column(waveform_plot, xcorr_plot), Spacer(width=30),
            column(buffer_slider, connect_toggle, reset_button)),
        row(pulse_delay_plot, Spacer(width=10), pulse_length_plot))

    return Panel(child=tab_stream_layout, title="Stream")
