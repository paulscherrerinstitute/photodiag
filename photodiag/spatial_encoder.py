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
        step_length: length of a step waveform in pix
    """

    def __init__(
            self, channel, roi=(200, 300), background_method='div', step_length=50,
            events_channel=None, refinement=1,
        ):
        """Initialize SpatialEncoder object.

        Args:
            channel: data channel of spatial encoder
            roi: region of interest for spatial encoder image projection along y-axis
            background_method: {'div', 'sub'} background removal method
                'div': data = data / background - 1
                'sub': data = data - background
            step_length: length of a step waveform in pix
        """
        self.channel = channel
        self.roi = roi
        self.background_method = background_method
        self.step_length = step_length
        self.refinement = refinement
        self.events_channel = events_channel
        self._background = None
        self._fs_per_pix = None

    def calibrate_background(self, filepath):
        """Calibrate spatial encoder background.

        Args:
            filepath: hdf5 file to be processed with background signal data
        """
        background_data, _, is_data_present, is_dark = self._read_bsread_file(filepath)
        if not any(is_data_present):
            raise Exception("is_data_present is 0 for all pulses in {}".format(self.channel))

        # average over all dark images with data being present
        if is_dark is None:
            filter_ind = is_data_present
        else:
            filter_ind = np.logical_and(is_data_present, is_dark)
        self._background = background_data[filter_ind].mean(axis=0)

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
                data, _, is_data_present, _ = self._read_bsread_file(bsread_file)
                data = data[is_data_present].mean(axis=0)

                results = self.process(data)
                edge_pos_pix[i] = results['egde_pos']

        elif method == 'avg_edge':
            results = self.process_eco(filepath)

            scan_pos_fs = np.empty(len(results))
            edge_pos_pix = np.empty(len(results))
            for i, data in enumerate(results):
                scan_pos_fs[i] = data['scan_pos_fs']
                edge_pos_pix[i] = data['edge_pos'].nanmean()

        # pixel -> fs conversion coefficient
        fit_coeff = np.polyfit(edge_pos_pix, scan_pos_fs, 1)
        self._fs_per_pix = fit_coeff[0]

    def process(self, data, is_dark=None, debug=False):
        """Process spatial encoder data.

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

        # transform vector to array for consistency
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.ndim > 2:
            raise Exception('Input data should be either 1- or 2-dimentional array')

        # remove background
        if self.background_method == 'sub':
            data -= self._background
        elif self.background_method == 'div':
            data /= self._background
            data -= 1
        else:
            raise Exception("Unknown background removal method '{}'".format(self.background_method))

        # refine data
        def _interp(fp, xp, x):  # utility function to be used with apply_along_axis
            return np.interp(x, xp, fp)

        data_length = data.shape[1]
        refined_data = np.apply_along_axis(
            _interp, 1, data,
            x=np.arange(0, data_length-1, self.refinement),
            xp=np.arange(data_length),
        )

        # prepare a step function and refine it
        step_waveform = np.ones(shape=(self.step_length, ))
        step_waveform[:int(self.step_length/2)] = -1

        step_waveform = np.interp(
            x=np.arange(0, self.step_length-1, self.refinement),
            xp=np.arange(self.step_length),
            fp=step_waveform,
        )

        # find edges
        xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode='valid')
        edge_position = np.argmax(xcorr, axis=1).astype(float) * self.refinement
        xcorr_amplitude = np.amax(xcorr, axis=1)

        # correct edge_position for step_length
        edge_position += np.floor(self.step_length/2)

        if is_dark is not None:
            edge_position[is_dark] = np.nan

        output = {'edge_pos': edge_position, 'xcorr_ampl': xcorr_amplitude}

        if debug:
            output['xcorr'] = xcorr
            output['raw_input'] = data

        return output

    def process_hdf5(self, filepath, debug=False):
        """Process spatial encoder data from hdf5 file.

        Args:
            filepath: hdf5 file to be processed
            debug: return debug data
        Returns:
            edge position(s) in pix and corresponding pulse ids
            cross-correlation results and raw data if `debug` is True
        """
        if self.events_channel:
            self.calibrate_background(filepath)
        else:
            if self._background is None:
                raise Exception("Background calibration is not found")

        data, pulse_id, is_data_present, is_dark = self._read_bsread_file(filepath)
        output = self.process(data, debug=debug, is_dark=is_dark)

        output['edge_pos'][~is_data_present] = np.nan
        output['pulse_id'] = pulse_id

        return output

    def process_eco(self, filepath, debug=False):
        """Process spatial encoder data from eco scan file.

        Args:
            filepath: json eco scan file to be processed
            debug: return debug data
        Returns:
            edge position(s) in pix, corresponding pulse ids and scan readback values
            cross-correlation results and raw data if `debug` is True
        """
        if self.events_channel:
            pass
        else:
            if self._background is None:
                raise Exception("Background calibration is not found")

        scan_pos_fs, bsread_files = self._read_eco_scan(filepath)

        output = []
        for i, bsread_file in enumerate(bsread_files):
            step_output = self.process_hdf5(bsread_file, debug=debug)
            step_output['scan_pos_fs'] = scan_pos_fs[i]
            output.append(step_output)

        return output

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

            if self.events_channel:
                events_channel_group = h5f["/data/{}".format(self.events_channel)]
                index = pulse_id - events_channel_group["pulse_id"][0]
                is_dark = events_channel_group["data"][index, 25].astype(bool)
            else:
                is_dark = None

        return data, pulse_id, is_data_present, is_dark

    def _read_bsread_image(self, filepath):
        """Read spatial encoder images from bsread hdf5 file.

        Args:
            filepath: path to a bsread hdf5 file to read data from
        Returns:
            data
        """
        with h5py.File(filepath, 'r') as h5f:
            channel_group = h5f["/data/{}".format(self.channel)]

            # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
            data = channel_group["data"][:, slice(*self.roi), :].astype(float)

        return data

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
