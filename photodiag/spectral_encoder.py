import warnings
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np

from .utils import find_edge, read_eco_scan

edge_types = ["falling", "rising"]


class SpectralEncoder:
    def __init__(
        self,
        signal_channel,
        background_channel,
        step_length=50,
        events_channel=None,
        dark_shot_event=21,
        early_shot_event=22,
        dark_shot_filter=None,
        early_shot_filter=None,
        refinement=1,
        edge_type="falling",
    ):
        """Initialize SpectralEncoder object.

        Args:
            signal_channel: signal data channel of spectral encoder
            background_channel: background data channel of spectral encoder
            step_length: length of a step waveform in pix
            events_channel: data channel of events
            dark_shot_event: event number for dark shots if events_channel is present
            early_shot_event: event number for early shots if events_channel is present
            dark_shot_filter: a function to return True for dark shots based on pulse_id argument
            early_shot_filter: a function to return True for early shots based on pulse_id argument
            refinement: quantisation size for linear interpolation of data and a step waveform
            edge_type: {'falling', 'rising'} a type of edge to search for
        """
        if events_channel and dark_shot_filter:
            raise Exception("Either 'events_channel' and/or 'dark_shot_filter' should be None")

        self.signal_channel = signal_channel
        self.background_channel = background_channel
        self.step_length = step_length
        self.refinement = refinement
        self.events_channel = events_channel
        self.dark_shot_event = dark_shot_event
        self.early_shot_event = early_shot_event
        self.dark_shot_filter = dark_shot_filter
        self.early_shot_filter = early_shot_filter
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
            raise ValueError(f"A reasonable step length should be >= 4")
        self.__step_length = value

    def calibrate_time(self, filepath, method="avg_edge", nproc=1):
        """Calibrate pixel to time conversion.

        Args:
            filepath: eco scan file to be used for pixel to femtosecond calibration
            method: {avg_wf, avg_edge}
                'avg_wf': single edge position of averaged raw waveform (per scan step)
                'avg_edge': mean of edge positions for all raw waveforms (per scan step)
            nproc: number of worker processes to use
        """
        if (
            self.events_channel is None
            and self.dark_shot_filter is None
            and self._background is None
        ):
            raise Exception("Background calibration is not found")

        if method == "avg_wf":
            scan_pos_fs, bsread_files = read_eco_scan(filepath)

            edge_pos_pix = np.empty(len(scan_pos_fs))
            for i, bsread_file in enumerate(bsread_files):
                data, _, _ = self._read_bsread_file(bsread_file)
                data = data.mean(axis=0)

                results = self.process(data)
                edge_pos_pix[i] = results["edge_pos"]

        elif method == "avg_edge":
            results = self.process_eco(filepath, nproc=nproc)

            scan_pos_fs = np.empty(len(results))
            edge_pos_pix = np.empty(len(results))
            for i, data in enumerate(results):
                scan_pos_fs[i] = data["scan_pos_fs"]
                edge_pos_pix[i] = np.nanmean(data["edge_pos"])

        # pixel -> fs conversion coefficient
        fit_coeff = np.polyfit(scan_pos_fs, edge_pos_pix, 1)
        self.pix_per_fs = fit_coeff[0]

        return scan_pos_fs, edge_pos_pix, fit_coeff

    def process(self, data, debug=False):
        """Process spectral encoder data.

        Edge detection is performed by finding a maximum of cross-convolution between a step
        profile and input data waveforms.

        Args:
            data: data to be processed
            debug: return debug data
        Returns:
            edge position(s) in pix
            cross-correlation results and raw data if `debug` is True
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

        if debug:
            output["raw_input"] = data

        return output

    def process_hdf5(self, filepath, debug=False):
        """Process spectral encoder data from hdf5 file.

        Args:
            filepath: hdf5 file to be processed
            debug: return debug data
        Returns:
            edge position(s) in pix and corresponding pulse ids
            cross-correlation results and raw data if `debug` is True
        """
        data, pulse_id, is_dark = self._read_bsread_file(filepath)

        output = self.process(data, debug=debug)

        if is_dark is not None:
            output["edge_pos"][is_dark] = np.nan

        output["pulse_id"] = pulse_id
        output["is_dark"] = is_dark

        return output

    def process_eco(self, filepath, nproc=1, debug=False):
        """Process spectral encoder data from eco scan file.

        Args:
            filepath: json eco scan file to be processed
            nproc: number of worker processes to use
            debug: return debug data
        Returns:
            edge position(s) in pix, corresponding pulse ids and scan readback values
            cross-correlation results and raw data if `debug` is True
        """
        if self.events_channel or self.dark_shot_filter:
            pass
        else:
            if self._background is None:
                raise Exception("Background calibration is not found")

        scan_pos_fs, bsread_files = read_eco_scan(filepath)

        with Pool(processes=nproc) as pool:
            output = pool.map(partial(self.process_hdf5, debug=debug), bsread_files)

        for i, step_output in enumerate(output):
            step_output["scan_pos_fs"] = scan_pos_fs[i]

        return output

    def _read_bsread_file(self, filepath):
        """Read spectral encoder data from bsread hdf5 file.

        Args:
            filepath: path to a bsread hdf5 file to read data from
        Returns:
            data, pulse_id, is_dark
        """
        with h5py.File(filepath, "r") as h5f:
            if "/data" in h5f:
                # sf_databuffer_writer format
                path_prefix = "/data/{}"
            else:
                # bsread format
                path_prefix = "/{}"

            signal_channel_group = h5f[path_prefix.format(self.signal_channel)]
            signal_pulse_id = signal_channel_group["pulse_id"][:]

            if self.events_channel:
                events_channel_group = h5f[path_prefix.format(self.events_channel)]
                events_pulse_id = events_channel_group["pulse_id"][:]

                pid, index, event_index = np.intersect1d(
                    signal_pulse_id, events_pulse_id, return_indices=True
                )

                # if both groups have 0 in their pulse_id
                pid_zero_ind = pid == 0
                if any(pid_zero_ind):
                    warnings.warn(
                        f"\n \
                    File: {filepath}\n \
                    Both '{self.signal_channel}' and '{self.events_channel}' have zeroed pulse_id(s).\n"
                    )
                    index = index[~pid_zero_ind]
                    event_index = event_index[~pid_zero_ind]

                is_dark = events_channel_group["data"][event_index, self.dark_shot_event].astype(
                    bool
                )

            elif self.dark_shot_filter:
                index = signal_pulse_id != 0
                is_dark = self.dark_shot_filter(signal_pulse_id)[index]

            else:
                index = signal_pulse_id != 0
                is_dark = None

            signal_pulse_id = signal_pulse_id[index]

            # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
            images = signal_channel_group["data"][index].astype(float)

            # averaging every image over y-axis gives the final raw waveforms
            data = images.mean(axis=1)

        return data, signal_pulse_id, is_dark
