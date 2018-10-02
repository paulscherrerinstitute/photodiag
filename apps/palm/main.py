from bokeh.io import curdoc
from bokeh.models import Tabs

from panel_calib import tab_calibration
from panel_h5file import tab_hdf5file
from panel_stream import tab_stream

doc = curdoc()
doc.title = "PALM"

# Final layout
doc.add_root(Tabs(tabs=[tab_calibration, tab_hdf5file, tab_stream]))
