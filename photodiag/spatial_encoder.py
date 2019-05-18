import json
import warnings
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np


class SpatialEncoder:
    def __init__(
            self, channel, roi=(200, 300), background_method='div', step_length=50,
            events_channel=None, dark_shot_event=21, refinement=1,
        ):
        """Initialize SpatialEncoder object.

        Args:
            channel: data channel of spatial encoder
            roi: region of interest for spatial encoder image projection along y-axis
            background_method: {'div', 'sub'} background removal method
                'div': data = data / background - 1
                'sub': data = data - background
            step_length: length of a step waveform in pix
            events_channel: data channel of events
            refinement: quantisation size for linear interpolation of data and a step waveform
        """
        self.channel = channel
        self.roi = roi
        self.background_method = background_method
        self.step_length = step_length
        self.refinement = refinement
        self.events_channel = events_channel
        self.dark_shot_event = dark_shot_event
        self._background = None
        self.pix_per_fs = None

    def calibrate_background(self, background_data, is_dark):
        """Calibrate spatial encoder background.

        Args:
            background_data: array with background signal data
            is_dark: index of dark shots
        """
        # average over all dark images
        if is_dark is None:
            self._background = background_data.mean(axis=0)
        else:
            self._background = background_data[is_dark].mean(axis=0)

    def calibrate_time(self, filepath, method='avg_edge', nproc=1):
        """Calibrate pixel to time conversion.

        Args:
            filepath: eco scan file to be used for pixel to femtosecond calibration
            method: {avg_wf, avg_edge}
                'avg_wf': single edge position of averaged raw waveform (per scan step)
                'avg_edge': mean of edge positions for all raw waveforms (per scan step)
        """
        if self.events_channel is None and self._background is None:
            raise Exception("Background calibration is not found")

        if method == 'avg_wf':
            scan_pos_fs, bsread_files = self._read_eco_scan(filepath)

            edge_pos_pix = np.empty(len(scan_pos_fs))
            for i, bsread_file in enumerate(bsread_files):
                data, _, _, _ = self._read_bsread_file(bsread_file)
                data = data.mean(axis=0)

                results = self.process(data)
                edge_pos_pix[i] = results['edge_pos']

        elif method == 'avg_edge':
            results = self.process_eco(filepath, nproc=nproc)

            scan_pos_fs = np.empty(len(results))
            edge_pos_pix = np.empty(len(results))
            for i, data in enumerate(results):
                scan_pos_fs[i] = data['scan_pos_fs']
                edge_pos_pix[i] = np.nanmean(data['edge_pos'])

        # pixel -> fs conversion coefficient
        fit_coeff = np.polyfit(scan_pos_fs, edge_pos_pix, 1)
        self.pix_per_fs = fit_coeff[0]

        return scan_pos_fs, edge_pos_pix, fit_coeff

    def process(self, data, debug=False):
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
        data, pulse_id, is_dark, images = self._read_bsread_file(filepath, return_images=debug)

        if self.events_channel:
            self.calibrate_background(data, is_dark)
        else:
            if self._background is None:
                raise Exception("Background calibration is not found")

        output = self.process(data, debug=debug)

        if is_dark is not None:
            output['edge_pos'][is_dark] = np.nan

        output['pulse_id'] = pulse_id
        output['images'] = images

        return output

    def process_eco(self, filepath, nproc=1, debug=False):
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

        with Pool(processes=nproc) as pool:
            output = pool.map(partial(self.process_hdf5, debug=debug), bsread_files)

        for i, step_output in enumerate(output):
            step_output['scan_pos_fs'] = scan_pos_fs[i]

        return output

    def _read_bsread_file(self, filepath, return_images=False):
        """Read spatial encoder data from bsread hdf5 file.

        Args:
            filepath: path to a bsread hdf5 file to read data from
        Returns:
            data, pulse_id, is_dark
        """
        with h5py.File(filepath, 'r') as h5f:
            if "/data" in h5f:
                # sf_databuffer_writer format
                path_prefix = "/data/{}"
            else:
                # bsread format
                path_prefix = "/{}"

            channel_group = h5f[path_prefix.format(self.channel)]
            pulse_id = channel_group["pulse_id"][:]

            if self.events_channel:
                events_channel_group = h5f[path_prefix.format(self.events_channel)]
                events_pulse_id = events_channel_group["pulse_id"][:]

                pid, index, event_index = np.intersect1d(
                    pulse_id, events_pulse_id, return_indices=True,
                )

                # if both groups have 0 in their pulse_id
                pid_zero_ind = pid == 0
                if any(pid_zero_ind):
                    warnings.warn(f"\n \
                    File: {filepath}\n \
                    Both '{self.channel}' and '{self.events_channel}' have zeroed pulse_id(s).\n")
                    index = index[~pid_zero_ind]
                    event_index = event_index[~pid_zero_ind]

                is_dark = events_channel_group["data"][event_index, self.dark_shot_event] \
                    .astype(bool)

            else:
                index = pulse_id != 0
                is_dark = None

            pulse_id = pulse_id[index]

            # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
            images = channel_group["data"][index, slice(*self.roi), :].astype(float)

            # averaging every image over y-axis gives the final raw waveforms
            data = images.mean(axis=1)

        if return_images:
            return data, pulse_id, is_dark, images

        return data, pulse_id, is_dark, None

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
