import json

import h5py
import numpy as np


class SpatialEncoder:
    """Class describing spatial encoder setup.

    Attributes:
        channel: data channel of spatial encoder
        roi: region of interest for spatial encoder image projection along y-axis
    """

    def __init__(self, channel, roi=(200, 300)):
        """Initialize SpatialEncoder object.

        Args:
            channel: data channel of spatial encoder
            roi: region of interest for spatial encoder image projection along y-axis
        """
        self.channel = channel
        self.roi = roi
        self._background = None

    def calibrate_background(self, filepath):
        """Calibrate spatial encoder.

        Args:
            filepath: hdf5 file to be processed with background signal data
        """
        background_data, _, is_data_present = self._read_bsread_file(filepath)

        # average over all images with data being present
        self._background = background_data[is_data_present].mean(axis=0)

    def process(self, data, step_length=50, debug=False):
        """Process spatial encoder data.

        Edge detection is performed by finding a maximum of cross-convolution between a step
        profile and input data waveforms.

        Args:
            data: data to be processed
            step_length: length of a step waveform in pix
            debug: return debug data
        Returns:
            edge position(s) in pix
            cross-correlation results and raw data if `debug` is True
        """
        if self._background is None:
            raise Exception("Background calibration is not found")

        # remove background
        data /= self._background

        # prepare a step function
        step_waveform = np.ones(shape=(step_length, ))
        step_waveform[:int(step_length/2)] = -1

        # broadcast cross-correlation function in case of a 2-dimentional array
        if data.ndim == 1:
            xcorr = np.correlate(data, step_waveform, mode='valid')
            edge_position = np.argmax(xcorr).astype(float)
        elif data.ndim == 2:
            xcorr = np.apply_along_axis(np.correlate, 1, data, step_waveform, mode='valid')
            edge_position = np.argmax(xcorr, axis=1).astype(float)
        else:
            raise Exception('Input data should be either 1- or 2-dimentional array')

        # correct edge_position for step_length
        edge_position += step_length/2

        if debug:
            output = edge_position, xcorr, data
        else:
            output = edge_position

        return output

    def process_hdf5(self, filepath, step_length=50, debug=False):
        """Process spatial encoder data from hdf5 file.

        Args:
            filepath: hdf5 file to be processed
            step_length: length of a step waveform in pix
            debug: return debug data
        Returns:
            edge position(s) in pix and corresponding pulse ids
            cross-correlation results and raw data if `debug` is True
        """
        if self._background is None:
            raise Exception("Background calibration is not found")

        data, pulse_id, is_data_present = self._read_bsread_file(filepath)
        output = self.process(data[is_data_present], step_length=step_length, debug=debug)

        return output, pulse_id[is_data_present]

    def process_eco(self, filepath, step_length=50, debug=False):
        """Process spatial encoder data from eco scan file.

        Args:
            filepath: json eco scan file to be processed
            step_length: length of a step waveform in pix
            debug: return debug data
        Returns:
            edge position(s) in pix, corresponding pulse ids and scan readback values
            cross-correlation results and raw data if `debug` is True
        """
        with open(filepath) as eco_file:
            eco_scan = json.load(eco_file)

        scan_readbacks = eco_scan['scan_readbacks']

        output = []
        pulse_id = []
        for scan_files in eco_scan['scan_files']:
            # bsread file is 'normally' a first file on a list, but maybe the following should be
            # implemented in a more robust way
            bsread_file = scan_files[0]
            out, pid = self.process_hdf5(bsread_file, step_length=step_length, debug=debug)

            output.append(out)
            pulse_id.append(pid)

        return output, pulse_id, scan_readbacks

    def _read_bsread_file(self, filepath):
        """Read spatial encoder data from bsread hdf5 file.

        Args:
            filepath: path to a bsread hdf5 file to read data from
        Returns:
            data, pulse_id, is_data_present
        """
        with h5py.File(filepath, 'r') as h5f:
            channel_group = h5f["/data/{}".format(self.channel)]

            is_data_present = channel_group["is_data_present"][:]
            if not any(is_data_present):
                raise Exception("is_data_present is 0 for all pulses in {}".format(self.channel))

            pulse_id = channel_group["pulse_id"][:]

            # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
            # averaging every image over y-axis gives the final raw waveforms
            data = channel_group["data"][:, slice(*self.roi), :].astype(float).mean(axis=1)

        return data, pulse_id, is_data_present
