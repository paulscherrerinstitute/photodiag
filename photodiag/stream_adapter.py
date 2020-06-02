import json

import numpy as np

from .utils import find_edge

edge_types = ["falling", "rising"]


class StreamAdapter:
    def __init__(
        self,
        json_config,
        step_length=50,
        refinement=1,
        edge_type="falling",
    ):
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

    def update_background(self, data):
        pass

    def process(self, data):
        """Process encoder data.

        Edge detection is performed by finding a maximum of cross-convolution between a step
        profile and input data waveforms.

        Args:
            data: data to be processed
        Returns:
            edge position(s) in pix
            cross-correlation results
        """
        if self._background is None:
            raise Exception("Background calibration is not found")

        if data.ndim == 1:
            # transform vector to array for consistency
            data = data[np.newaxis, :]
        elif data.ndim > 2:
            raise Exception("Input data should be either 1- or 2-dimentional array")

        # remove background
        data /= self._background
        np.log10(data, out=data)

        output = find_edge(data, self.step_length, self.edge_type, self.refinement)

        return output
