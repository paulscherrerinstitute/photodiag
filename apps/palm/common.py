import numpy as np

import photodiag
import receiver

# Create a palm setup object
palm = photodiag.PalmSetup(
    channels={'0': receiver.reference, '1': receiver.streaked},
    noise_range=[0, 250], energy_range=np.linspace(4850, 5150, 301))


# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
WAVEFORM_CANVAS_WIDTH = 620
WAVEFORM_CANVAS_HEIGHT = 380
