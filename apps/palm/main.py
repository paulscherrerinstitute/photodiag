import numpy as np
from bokeh.io import curdoc
from bokeh.models import Tabs

import photodiag
import receiver
import panel_calib
import panel_h5file
import panel_stream

doc = curdoc()
doc.title = "PALM"

# Create a palm setup object
palm = photodiag.PalmSetup(
    channels={'0': receiver.reference, '1': receiver.streaked},
    noise_range=[0, 250], energy_range=np.linspace(4850, 5150, 301))

# Final layout
tab_calib = panel_calib.create(palm)
tab_h5file = panel_h5file.create(palm)
tab_stream = panel_stream.create(palm)
doc.add_root(Tabs(tabs=[tab_calib, tab_h5file, tab_stream]))
