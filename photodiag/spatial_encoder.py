import json

import h5py
import numpy as np


class SpatialEncoder:
    """Class describing spatial encoder setup.

    Attributes:
        channel: data channel of spatial encoder
        roi: region of interest for spatial encoder image projection along y-axis
        background_method: {'div', 'sub'} background removal method
            'div': data = data / background - 1
            'sub': data = data - background
    """

    def __init__(self, channel, roi=(200, 300), background_method='div'):
        """Initialize SpatialEncoder object.

        Args:
            channel: data channel of spatial encoder
            roi: region of interest for spatial encoder image projection along y-axis
            background_method: {'div', 'sub'} background removal method
                'div': data = data / background - 1
                'sub': data = data - background
        """
        self.channel = channel
        self.roi = roi
        self.background_method = background_method
        self._background = None
        self._fs_per_pix = None

    def calibrate_background(self, filepath):
        """Calibrate spatial encoder background.

        Args:
            filepath: hdf5 file to be processed with background signal data
        """
        background_data, _, is_data_present = self._read_bsread_file(filepath)
        if not any(is_data_present):
            raise Exception("is_data_present is 0 for all pulses in {}".format(self.channel))

        # average over all images with data being present
        self._background = background_data[is_data_present].mean(axis=0)

    def calibrate_time(self, filepath, method='avg_edge'):
        """Calibrate pixel to time conversion.

        Args:
            filepath: eco scan file to be used for pixel to femtosecond calibration
            method: {avg_wf, avg_edge}
                'avg_wf': single edge position of averaged raw waveform (per scan step)
                'avg_edge': mean of edge positions for all raw waveforms (per scan step)
        """
        if self._background is None:
            raise Exception("Background calibration is not found")

        if method == 'avg_wf':
            scan_pos_fs, bsread_files = self._read_eco_scan(filepath)

            edge_pos_pix = np.empty(len(scan_pos_fs))
            for i, bsread_file in enumerate(bsread_files):
                data, _, is_data_present = self._read_bsread_file(bsread_file)
                data = data[is_data_present].mean(axis=0)
                edge_pos_pix[i] = self.process(data)

        elif method == 'avg_edge':
            results, _, scan_pos_fs = self.process_eco(filepath)

            edge_pos_pix = np.empty(len(scan_pos_fs))
            for i, data in enumerate(results):
                edge_pos_pix[i] = data.nanmean()

        # pixel -> fs conversion coefficient
        fit_coeff = np.polyfit(edge_pos_pix, scan_pos_fs, 1)
        self._fs_per_pix = fit_coeff[0]

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
        if self.background_method == 'sub':
            data -= self._background
        elif self.background_method == 'div':
            data /= self._background
            data -= 1
        else:
            raise Exception("Unknown background removal method '{}'".format(self.background_method))

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
        output = self.process(data, step_length=step_length, debug=debug)

        if debug:
            output[0][~is_data_present] = np.nan
        else:
            output[~is_data_present] = np.nan

        return output, pulse_id

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
        if self._background is None:
            raise Exception("Background calibration is not found")

        scan_pos_fs, bsread_files = self._read_eco_scan(filepath)

        output = []
        pulse_id = []
        for bsread_file in bsread_files:
            out, pid = self.process_hdf5(bsread_file, step_length=step_length, debug=debug)

            output.append(out)
            pulse_id.append(pid)

        return output, pulse_id, scan_pos_fs

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
            pulse_id = channel_group["pulse_id"][:]

            # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
            # averaging every image over y-axis gives the final raw waveforms
            data = channel_group["data"][:, slice(*self.roi), :].astype(float).mean(axis=1)

        return data, pulse_id, is_data_present

    @staticmethod
    def _read_eco_scan(filepath):
        """Extract `scan_readbacks` and corresponding bsread `scan_files` from an eco scan.

        Args:
            filepath: path to a json eco scan file to read data from
        Returns:
            scan_pos_fs, bsread_files
        """
        with open(filepath) as eco_file:
            eco_scan = json.load(eco_file)

        # flatten scan_readbacks array and convert values to femtoseconds
        scan_pos_fs = np.ravel(eco_scan['scan_readbacks']) * 1e15

        scan_files = eco_scan['scan_files']
        # bsread file is 'normally' a first file on a list, but maybe the following should be
        # implemented in a more robust way
        bsread_files = [scan_file[0] for scan_file in scan_files]

        return scan_pos_fs, bsread_files
