from bokeh.layouts import column
from bokeh.models import Panel, Spacer, Spinner, TextInput


def create(palm):
    # Reference etof channel
    def ref_etof_channel_textinput_callback(_attr, _old_value, new_value):
        palm.channels["0"] = new_value

    ref_etof_channel_textinput = TextInput(
        title="Reference eTOF channel:", value=palm.channels["0"]
    )
    ref_etof_channel_textinput.on_change("value", ref_etof_channel_textinput_callback)

    # Streaking etof channel
    def str_etof_channel_textinput_callback(_attr, _old_value, new_value):
        palm.channels["1"] = new_value

    str_etof_channel_textinput = TextInput(
        title="Streaking eTOF channel:", value=palm.channels["1"]
    )
    str_etof_channel_textinput.on_change("value", str_etof_channel_textinput_callback)

    # XFEL energy value spinner
    def xfel_energy_spinner_callback(_attr, old_value, new_value):
        if new_value > 0:
            palm.xfel_energy = new_value
        else:
            xfel_energy_spinner.value = old_value

    xfel_energy_spinner = Spinner(title="XFEL energy, eV:", value=palm.xfel_energy, step=0.1)
    xfel_energy_spinner.on_change("value", xfel_energy_spinner_callback)

    # Binding energy value spinner
    def binding_energy_spinner_callback(_attr, old_value, new_value):
        if new_value > 0:
            palm.binding_energy = new_value
        else:
            binding_energy_spinner.value = old_value

    binding_energy_spinner = Spinner(
        title="Binding energy, eV:", value=palm.binding_energy, step=0.1
    )
    binding_energy_spinner.on_change("value", binding_energy_spinner_callback)

    # Zero drift tube value spinner
    def zero_drift_spinner_callback(_attr, old_value, new_value):
        if new_value > 0:
            palm.zero_drift_tube = new_value
        else:
            zero_drift_spinner.value = old_value

    zero_drift_spinner = Spinner(title="Zero drift tube, eV:", value=palm.zero_drift_tube, step=0.1)
    zero_drift_spinner.on_change("value", zero_drift_spinner_callback)

    tab_layout = column(
        ref_etof_channel_textinput,
        str_etof_channel_textinput,
        Spacer(height=30),
        xfel_energy_spinner,
        binding_energy_spinner,
        zero_drift_spinner,
    )

    return Panel(child=tab_layout, title="Setup")
