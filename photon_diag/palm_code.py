import os
import re

import h5py
import numpy as np
# TODO: consider to use pandas for certain operations
# import pandas as pd

from photon_diag.psen_code import PsenSetup
from photon_diag.spectrometer import Spectrometer


class PalmSetup:
    """Class describing the photon arrival and length monitor (PALM) setup.
    """

    def __init__(self, home_dir):
        """Initialize PALM setup object and optionally restore a particular state from the past.

        For the spectrometers the following notation is used: presense of a streaking field ('0': no
        streaking, '1': positive streaking, '-1': negative streaking)

        Args:
            home_dir: home directory for an experiment (location for all data files to be processed/derived)
        """
        self.home_dir = home_dir

        self.spectrometers = {'1': Spectrometer(path='C1'),
                              '0': Spectrometer(path='C3')}

        self.hdf5_range = [0, 4000]
        self.tags = []
        # self.energy_range = [8600, 9400]

    def calibrate(self, folder_name, bkg_en=None, etofs=None, overwrite=True):
        """General routine for a calibration process of the electron time of flight (eTOF) etofs.

        Args:
            folder_name: location of hdf5 files with calibration data
            bkg_en: (optional) background energy profile to be subtracted from other waveforms (e.g. to
                    remove influence of Auger peaks)
            etofs: (optional) list of etof spectrometers to be calibrated
            overwrite: (optional) start over a calibration process

        Returns:
            results of calibration as dictionary
        """
        calib_folder = self.home_dir + folder_name

        if etofs is None:
            calibrated_etofs = self.spectrometers.values()
        else:
            calibrated_etofs = etofs  # TODO: is there even need to calibrate etofs separately?

        if overwrite:
            for etof in calibrated_etofs:
                etof.calib_data = {}

        with os.scandir(calib_folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    energy = self._get_energy_from_filename(entry.path)

                    for etof in calibrated_etofs:
                        if not overwrite and energy in etof.calib_data.keys():
                            continue

                        # TODO: Ask Pavle about the etof internal time
                        etof.internal_time = self._get_internal_time(entry.path, etof.path, *self.hdf5_range)
                        _, calib_waveforms = self._get_tags_and_data(entry.path, etof.path, *self.hdf5_range)
                        etof.add_calibration_point(energy, calib_waveforms)

        calib_results = {}
        for etof_key in self.spectrometers.keys():
            calib_results[etof_key] = self.spectrometers[etof_key].fit_calibration_curve(bkg_en=bkg_en)

        return calib_results

    # TODO: process streaming data
    def process_hdf5_file(self, filename):
        """Load data for all registered spectrometers from an hdf5 file. This method is to be changed
        in order to adapt to a format of PALM data files in the future.

        Args:
            filename: file name to be loaded

        Returns:
            tuple of tags and the corresponding results in a dictionary
        """
        data_raw = {}
        time_raw = {}
        filepath = self.home_dir + filename

        for etof_key, etof in self.spectrometers.items():
            self.tags, data = self._get_tags_and_data(filepath, etof.path, *self.hdf5_range)
            data_raw[etof_key] = data
            # data_raw[etof_key] = np.expand_dims(data[1, :], axis=0)
            time_raw[etof_key] = self._get_internal_time(filepath, etof.path, *self.hdf5_range)

        results, prep_data = self.process(data_raw)

        return results, prep_data

    def process(self, waveforms, method='xcorr', jacobian=False, noise_thr=7):
        """Main function to analyse PALM data that pipelines separate stages of data processing.

        Args:
            waveforms: dictionary with waveforms from streaked and non-streaked spectrometers
            method: (optional) {'xcorr' (default), 'deconv'}
            jacobian: (optional) apply jacobian corrections of spectrometer's time to energy transformation
            noise_thr:

        Returns:
            pulse lengths and arrival times per pulse
        """
        prep_data = {}
        if method == 'xcorr':
            for etof_key, data in waveforms.items():
                etof = self.spectrometers[etof_key]
                # TODO: it can be ok to detect photon peaks from bulk data for a calibration check
                # self._detect_photon_peaks()
                prep_data[etof_key] = etof.prepare(data, jacobian=jacobian, noise_thr=noise_thr)

            results = self._cross_corr_analysis(prep_data)

        elif method == 'deconv':
            # prepare data
            results = self._deconvolution_analysis(prep_data)

        else:
            raise RuntimeError(f"Method '{method}' is not recognised")

        return results, prep_data

    @staticmethod
    def _get_internal_time(filepath, etof_path, first_ind=None, last_ind=None):
        """Read PALM internal time from an hdf5 file.

        Args:
            filepath: path to an hdf5 file
            etof_path: location of data in hdf5 file
            first_ind: (optional) index of a first element to read
            last_ind: (optional) index of a last element to read

        Returns:
            time: internal electron time-of-flight reference.
        """
        with h5py.File(filepath) as h5f:
            time = h5f[f'/{etof_path}/time'][first_ind:last_ind] * 1e9  # convert to fs

        return time

    @staticmethod
    def _get_tags_and_data(filepath, etof_path, first_ind=None, last_ind=None):
        """Read PALM waveforms from an hdf5 file for unique tags (ignoring data for all repeated tags).

        Args:
            filepath: path to an hdf5 file
            etof_path: location of data in hdf5 file
            first_ind: (optional) index of a first element to read
            last_ind: (optional) index of a last element to read

        Returns:
            tags and data
        """
        with h5py.File(filepath) as h5f:
            tags = h5f['/tags'][:]

            # eliminate repeated tags
            _, ind, counts = np.unique(tags, return_index=True, return_counts=True)
            idx = ind[counts == 1]
            tags = tags[idx]

            data = -h5f[f'/{etof_path}/data'][idx, first_ind:last_ind]

        return tags, data

    @staticmethod
    def _get_energy_from_filename(filename):
        """Parse filename and return energy value (first float number encountered). This method is likely to
        be changed in order to adapt to a format of PALM callibration files in the future.

        Args:
            filename: file name to be parsed

        Returns:
            energy in eV as a float number
        """
        energy = float(re.findall('\d+\.\d+', filename)[0])

        return 1000*energy  # convert to eV

    def _cross_corr_analysis(self, input_data):
        """Perform analysis to determine arrival times via cross correlation.

        Usually, this data can be used to initally identify pulses that are falling within linear slope of
        a THz streak pulse.

        Args:
            input_data: input data to be correlated

        Returns:
            pulse arrival delays via cross-correlation method
        """
        # TODO: generalize for different streaking field phases (=etof_key)
        data_str = input_data['1']
        data_nonstr = input_data['0']

        corr_results = np.empty_like(data_nonstr)
        for i, (x, y) in enumerate(zip(data_nonstr, data_str)):
            corr_results[i, :] = np.correlate(x, y, mode='same')

        corr_results = self._truncate_largest_peak(corr_results, 0)

        size = corr_results.shape[1]
        lags = np.array(range(-int(size/2), int(size/2)))

        delays, _ = self._peak_params(lags, corr_results)

        pulse_lengths = self._peak_center_of_mass(input_data, lags)

        return lags, delays, pulse_lengths

    def _deconvolution_analysis(self, input_data, iterations=200):
        """Perform analysis to determine temporal profile of photon pulses.

        Args:
            input_data: data to be analysed
            iterations: number of iterations for the deconvolution analysis

        Returns:
            result(s) of deconvolution
        """
        # TODO: generalize for different streaking field phases (=etof_key)
        data_str = input_data['1']
        data_nonstr = input_data['0']

        deconv_result = np.empty_like(data_str)
        for i, (x, y) in enumerate(zip(data_nonstr, data_str)):
            deconv_result[i] = richardson_lucy_deconv(x, y, iterations=iterations)

        return deconv_result

    @staticmethod
    def _peak_params(x, y):
        """Calculate discrete waveform's peak mean and variance.

        Waveform's mean ensures that: sum[(x_i - mean) * y_i] = 0, with i = 1...N
        Waveform's var ensures that: sum[((x_i - mean)^2 - var) * y_i] = 0, with i = 1...N

        Args:
            x: abscissas of a descrite waveform coordinates
            y: ordinates of a descrite waveform coordinates

        Returns:
            peak_mean, peak_var: mean and variance values
        """
        denom = np.sum(y, axis=1)
        denom[denom == 0] = 1  # TODO: fixit
        peak_mean = np.sum(x * y, axis=1) / denom
        peak_var = np.sum((x - peak_mean[:, np.newaxis])**2 * y, axis=1) / denom

        return peak_mean, peak_var

    @staticmethod
    def _truncate_largest_peak(y, thr):
        """Truncate the largest peak above a specified threshold.

        Args:
            y: waveform/graph to be truncated
            thr: threshold level in ordinate units

        Returns:
            ordinates of a truncated waveform
        """
        def test_fun(y_1d):
            y_above_thr = (y_1d > thr).astype(int)
            y_above_thr = np.pad(y_above_thr, (1,), 'constant')
            inout = np.diff(y_above_thr)

            ind_in = np.argwhere(inout == 1).flatten()
            ind_out = np.argwhere(inout == -1).flatten()

            if ind_in.size == 0:
                y_1d[:] = 0

            else:
                ind_max_length = np.argmax(ind_out - ind_in)

                y_1d[:ind_in[ind_max_length]] = 0
                y_1d[ind_out[ind_max_length]:] = 0

        if y.ndim == 1:
            test_fun(y)
        else:
            np.apply_along_axis(test_fun, 1, y)

        return y

    def _peak_center_of_mass(self, input_data, lags):
        """Estimate pulse lengths based on a peak's center of mass (COM) method.

        Args:
            input_data: data to be processed
            lags: time delays in arbitrary units

        Returns:
            pulse lenghts
        """
        data_str = input_data['1'].copy()
        data_nonstr = input_data['0'].copy()

        thr1 = np.mean(self.spectrometers['1'].noise_std)
        thr3 = np.mean(self.spectrometers['0'].noise_std)

        data_str = self._truncate_largest_peak(data_str, 0)
        data_nonstr = self._truncate_largest_peak(data_nonstr, 0)

        _, var1 = self._peak_params(lags, data_str)
        _, var3 = self._peak_params(lags, data_nonstr)

        ind = np.logical_and(~np.isnan(var1), ~np.isnan(var3))

        pulse_length = np.real(np.lib.scimath.sqrt(var1[ind] - var3[ind]))

        return pulse_length


def richardson_lucy_deconv(streaked_signal, base_signal, iterations=200, noise=0.3):
    """Deconvolve eTOF waveforms using Richardson-Lucy algorithm, extracting pulse profile in a time domain.

    The assumption is that the streaked waveform was created by convolving an with a point-spread function
    PSF and possibly by adding noise.

    Args:
        streaked_signal: waveform after streaking
        base_signal: waveform without effect of streaking
        iterations: number of Richardson-Lucy algorithm iterations
        noise: noise level in the units of waveform intensity

    Returns:
        pulse profile in a time domain
    """
    from numpy import conjugate, real, ones
    from numpy.fft import fft, ifft, fftshift

    # TODO: check the implementation of 'weight' parameter, for now keep it == 1
    # TODO: implement stability enforcement
    weight = ones(streaked_signal.shape)

    otf = fft(fftshift(base_signal))  # optical transfer function
    time_profile = streaked_signal.copy()
    time_profile.clip(min=0)

    weighted_signal = streaked_signal.copy() + noise
    weighted_signal = weight * weighted_signal.clip(min=0)

    scale = real(ifft(conjugate(otf)*fft(weight)))

    for _ in range(iterations):
        relative_psf = weighted_signal / (real(ifft(otf*fft(time_profile))) + noise)
        time_profile *= real(ifft(conjugate(otf)*fft(relative_psf))) / scale
        time_profile.clip(min=0)

    return time_profile
