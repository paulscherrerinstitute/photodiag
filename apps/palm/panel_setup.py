from bokeh.layouts import column
from bokeh.models import Panel, Spacer, TextInput


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


    # XFEL energy value text input
    def xfel_energy_textinput_callback(_attr, old, new):
        try:
            new_value = float(new)
            if new_value > 0:
                palm.xfel_energy = new_value
            else:
                xfel_energy_textinput.value = old

        except ValueError:
            xfel_energy_textinput.value = old

    xfel_energy_textinput = TextInput(title='XFEL energy, eV:', value=str(palm.xfel_energy))
    xfel_energy_textinput.on_change('value', xfel_energy_textinput_callback)


    # Binding energy value text input
    def binding_energy_textinput_callback(_attr, old, new):
        try:
            new_value = float(new)
            if new_value > 0:
                palm.binding_energy = new_value
            else:
                binding_energy_textinput.value = old

        except ValueError:
            binding_energy_textinput.value = old

    binding_energy_textinput = TextInput(
        title='Binding energy, eV:', value=str(palm.binding_energy))
    binding_energy_textinput.on_change('value', binding_energy_textinput_callback)


    # Zero drift tube value text input
    def zero_drift_textinput_callback(_attr, old, new):
        try:
            new_value = float(new)
            if new_value > 0:
                palm.zero_drift_tube = new_value
            else:
                zero_drift_textinput.value = old

        except ValueError:
            zero_drift_textinput.value = old

    zero_drift_textinput = TextInput(title='Zero drift tube, eV:', value=str(palm.zero_drift_tube))
    zero_drift_textinput.on_change('value', zero_drift_textinput_callback)


    tab_layout = column(
        ref_etof_channel_textinput, str_etof_channel_textinput,
        Spacer(height=30),
        xfel_energy_textinput, binding_energy_textinput, zero_drift_textinput,
        )

    return Panel(child=tab_layout, title="Setup")
