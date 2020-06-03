import json
from collections import deque

import numpy as np

from .utils import find_edge_1d, savgol_filter_1d

edge_types = ["falling", "rising"]

bkg_deque = deque(maxlen=5)
ref_deque = deque(maxlen=5)
ref_correction_deque = deque(maxlen=5)

I0_deque = deque(maxlen=500)
Xcor_deque = deque(maxlen=500)
Xcor_deque_ref = deque(maxlen=500)

savgol_period = 71
savgol_window = (368.45, 660.70)
savgol_steps = 2038


class StreamAdapter:
    def __init__(self, json_config, step_length=50, refinement=1, edge_type="falling"):
        """Initialize StreamAdapter object.

        Args:
            step_length: length of a step waveform in pix
            dark_shot_filter: a function to return True for dark shots based on pulse_id argument
            refinement: quantisation size for linear interpolation of data and a step waveform
            edge_type: {'falling', 'rising'} a type of edge to search for
        """
        with open(json_config) as f:
            self.config = json.load(f)

        self.step_length = step_length
        self.refinement = refinement
        self._background = None
        self.pix_per_fs = None
        self.edge_type = edge_type

    @property
    def edge_type(self):
        return self.__edge_type

    @edge_type.setter
    def edge_type(self, value):
        if value not in edge_types:
            raise ValueError(f"Unknown edge type '{value}'")
        self.__edge_type = value

    @property
    def step_length(self):
        return self.__step_length

    @step_length.setter
    def step_length(self, value):
        if value < 4:
            raise ValueError("A reasonable step length should be >= 4")
        self.__step_length = value

    def process(self, message, preproc_filter=True):
        """Process stream message.

        Edge detection is performed by finding a maximum of cross-convolution between a step
        profile and input data waveforms.

        Args:
            message: stream message to be processed
        """
        events = message.data.data[self.config["events"]].value

        if not events[self.config["laser"]]:
            # No laser, so skip updates to either background or signal
            return

        is_delayed = events[self.config["delayed"]].astype(bool)
        signal = message.data.data[self.config["ROI_signal"]].value
        ref = message.data.data[self.config["ROI_background"]].value

        if not is_delayed:
            I0_deque.append(message.data.data[self.config["I0"]].value)

        if preproc_filter:
            signal = savgol_filter_1d(signal, savgol_period, savgol_window, savgol_steps)
            ref = savgol_filter_1d(ref, savgol_period, savgol_window, savgol_steps)

        if is_delayed:  # update background (signal roi is a background)
            # TODO: can the ref be used as a background too?
            bkg_deque.append(signal)

        else:  # extract edge
            if bkg_deque:  # remove background
                signal_wo_bkg = signal / (sum(bkg_deque) / len(bkg_deque))
                res = find_edge_1d(signal_wo_bkg, self.step_length, self.edge_type)
                Xcor_deque.append(np.max(res["xcorr"][0]))

        ref_deque.append(ref)
        avg_ref = sum(ref_deque) / len(ref_deque)

        if is_delayed:  # update background
            signal_wo_ref = signal / avg_ref
            ref_correction_deque.append(signal_wo_ref)

        else:  # extract edge
            if ref_correction_deque:
                avg_ref /= (sum(ref_correction_deque) / len(ref_correction_deque))

            signal_wo_ref = signal / avg_ref
            res_ref = find_edge_1d(signal_wo_ref, self.step_length, self.edge_type)
            Xcor_deque_ref.append(np.max(res_ref["xcorr"][0]))
