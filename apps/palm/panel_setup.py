from bokeh.layouts import column
from bokeh.models import Panel, TextInput

def create(palm):
    # Reference etof channel
    def ref_etof_channel_textinput_callback(_attr, _old, new):
        palm.channels['0'] = new

    ref_etof_channel_textinput = TextInput(
        title='Reference eTOF channel:', value=palm.channels['0'])
    ref_etof_channel_textinput.on_change('value', ref_etof_channel_textinput_callback)

    # Streaking etof channel
    def str_etof_channel_textinput_callback(_attr, _old, new):
        palm.channels['1'] = new

    str_etof_channel_textinput = TextInput(
        title='Streaking eTOF channel:', value=palm.channels['1'])
    str_etof_channel_textinput.on_change('value', str_etof_channel_textinput_callback)

    tab_layout = column(ref_etof_channel_textinput, str_etof_channel_textinput)

    return Panel(child=tab_layout, title="Setup")
