import numpy as np
from bokeh.io import curdoc
from bokeh.models import Tabs

import photodiag
import receiver
import panel_etof_calib
import panel_thz_calib
import panel_h5file
import panel_stream
import panel_setup

doc = curdoc()
doc.title = "PALM"

# Create a palm setup object
palm = photodiag.PalmSetup(
    channels={'0': receiver.reference, '1': receiver.streaked},
    noise_range=[0, 250], energy_range=np.linspace(4850, 5150, 301))

# Final layout
tab_etof_calib = panel_etof_calib.create(palm)
tab_thz_calib = panel_thz_calib.create(palm)
tab_h5file = panel_h5file.create(palm)
tab_stream = panel_stream.create(palm)
tab_setup = panel_setup.create(palm)

doc.add_root(Tabs(tabs=[tab_setup, tab_etof_calib, tab_h5file, tab_thz_calib, tab_stream]))
