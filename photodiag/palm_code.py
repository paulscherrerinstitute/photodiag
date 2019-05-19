import datetime
import json
import logging
import os
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from photodiag.spectrometer import Spectrometer

log = logging.getLogger(__name__)


class PalmSetup:
    """Class describing the photon arrival and length monitor (PALM) setup.
    """

    def __init__(self, channels, noise_range, energy_range):
        """Initialize PALM setup object.

        For the electron time of flight (eTOF) spectrometers the following notation is used:
        presense of a streaking field ('0': no streaking (reference), '1': positive streaking,
        '-1': negative streaking)

        Args:
            noise_range: a range of spectrometer bins that does not contain any signals (currently,
                the same range will be applied for all spectrometers)
            energy_range: energy interpolation points (eV) to be used for convering
                spectrometer waveforms from 'time of flight' into 'energy' domain
        """
        self.channels = channels
        self.etofs = {'0': Spectrometer(noise_range), '1': Spectrometer(noise_range)}
        self.energy_range = energy_range

        self.thz_calib_data = pd.DataFrame(
            {
                'peak_shift': np.array([], dtype=float),
                'peak_shift_mean': np.array([], dtype=float),
                'peak_shift_std': np.array([], dtype=float),
            }
        )
        self.thz_slope = None
        self.thz_intersect = None
        self.thz_motor_name = None
        self.thz_motor_unit = None

        self.xfel_energy = 7085
        self.zero_drift_tube = 7500
        self.binding_energy = 1148.7

    def calibrate_etof(self, folder_name, etofs=None, overwrite=True):
        """General routine for a calibration process of the eTOF spectrometers.

        Args:
            folder_name: location of hdf5 files with calibration data
            etofs: (optional) list of eTOF spectrometer keys to be calibrated
            overwrite: (optional) start over a calibration process

        Returns:
            results of calibration as dictionary
        """
        if etofs is None:
            calibrated_etofs = self.etofs.keys()
        else:
            calibrated_etofs = etofs

        if overwrite:
            for etof_key in calibrated_etofs:
                etof = self.etofs[etof_key]
                etof.calib_data.drop(etof.calib_data.index[:], inplace=True)

        for entry in os.scandir(folder_name):
            if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                energy = (
                    self.xfel_energy - self.binding_energy - get_energy_from_filename(entry.name)
                )

                for etof_key in calibrated_etofs:
                    etof = self.etofs[etof_key]
                    if not overwrite and energy in etof.calib_data.index:
                        continue

                    _, calib_waveforms = get_tags_and_data(entry.path, self.channels[etof_key])

                    etof.add_calibration_point(energy, calib_waveforms)

        calib_results = {}
        for etof_key in self.etofs:
            calib_results[etof_key] = self.etofs[etof_key].fit_calibration_curve()

        return calib_results

    def calibrate_etof_eco(self, eco_scan_filename):
        with open(eco_scan_filename) as eco_scan:
            data = json.load(eco_scan)

        # Flatten scan_files list
        scan_files = [item for sublist in data['scan_files'] for item in sublist]
        scan_values = data['scan_values']

        for etof in self.etofs.values():
            etof.calib_data.drop(etof.calib_data.index[:], inplace=True)

        for scan_file, scan_value in zip(scan_files, scan_values):
            energy = scan_value[0]
            try:
                _, calib_waveforms0 = get_tags_and_data(scan_file, self.channels['0'])
                _, calib_waveforms1 = get_tags_and_data(scan_file, self.channels['1'])
            except Exception as e:
                log.warning(e)
            else:
                eff_bind_en = self.binding_energy + (self.zero_drift_tube - 1000 * energy)
                self.etofs['0'].add_calibration_point(eff_bind_en, calib_waveforms0)
                self.etofs['1'].add_calibration_point(eff_bind_en, calib_waveforms1)

        calib_results = {}
        for etof_key in self.etofs:
            calib_results[etof_key] = self.etofs[etof_key].fit_calibration_curve()

        return calib_results

    def save_etof_calib(self, path, file=None):
        """ Save eTOF calibration to a file.
        """
        if not file:
            file = str(datetime.datetime.now().isoformat(sep='_', timespec='seconds'))

        if not file.endswith('.palm_etof'):
            file += '.palm_etof'

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, file), 'wb') as f:
            pickle.dump(self.etofs, f)
            log.info("Save etof calibration to a file: %s", os.path.join(path, file))

    def load_etof_calib(self, filepath):
        """Load eTOF calibration from a file.
        """
        with open(filepath, 'rb') as f:
            self.etofs = pickle.load(f)
            log.info("Load etof calibration from a file: %s", filepath)

    def process(
        self, waveforms, method='xcorr', jacobian=False, noise_thr=0, debug=False, peak='max'
    ):
        """Main function to analyse PALM data that pipelines separate stages of data processing.

        Args:
            waveforms: dictionary with waveforms from streaked and non-streaked spectrometers
            method: (optional) currently, only one method is available {'xcorr' (default), 'deconv'}
            jacobian: (optional) apply jacobian corrections of spectrometer's time to energy
                transformation
            noise_thr:
            debug: (optional) return debug data

        Returns:
            pulse lengths and arrival times per pulse
        """
        prep_data = {}
        for etof_key, data in waveforms.items():
            etof = self.etofs[etof_key]
            prep_data[etof_key] = etof.convert(
                data, self.energy_range, jacobian=jacobian, noise_thr=noise_thr
            )

        if method == 'xcorr':
            results = self._cross_corr_analysis(prep_data, debug=debug, peak=peak)

        elif method == 'deconv':
            results = self._deconvolution_analysis(prep_data, debug=debug)

        else:
            raise RuntimeError("Method '{}' is not recognised".format(method))

        return results

    def calibrate_thz(self, path, fit_range=(-np.inf, np.inf)):
        """Calibrate THz pulse.
        """
        with open(path) as eco_scan:
            data = json.load(eco_scan)

        # Flatten lists
        scan_files = [item for sublist in data['scan_files'] for item in sublist]
        scan_readbacks = [item for sublist in data['scan_readbacks'] for item in sublist]

        # Convert to fs
        scan_readbacks = [1e15 * item for item in scan_readbacks]
        self.thz_motor_unit = 'fs'

        self.thz_motor_name = data['scan_parameters']['Id'][0]

        self.thz_calib_data.drop(self.thz_calib_data.index[:], inplace=True)
        for scan_file, scan_readback in zip(scan_files, scan_readbacks):
            try:
                _, peak_shift, _ = self.process_hdf5_file(scan_file)
            except Exception as e:
                log.warning(e)
            else:
                self.thz_calib_data.loc[scan_readback] = {
                    'peak_shift': peak_shift,
                    'peak_shift_mean': peak_shift.mean(),
                    'peak_shift_std': peak_shift.std(),
                }

        def fit_func(shift, a, b):
            return a * shift + b

        x_fit = self.thz_calib_data.index.values
        y_fit = self.thz_calib_data['peak_shift_mean'].values

        in_range = np.logical_and(fit_range[0] <= x_fit, x_fit <= fit_range[1])
        popt, _pcov = curve_fit(fit_func, x_fit[in_range], y_fit[in_range])

        self.thz_slope, self.thz_intersect = popt

    def save_thz_calib(self, path, file=None):
        """ Save THz pulse calibration to a file.
        """
        if not file:
            file = str(datetime.datetime.now().isoformat(sep='_', timespec='seconds'))

        if not file.endswith('.palm_thz'):
            file += '.palm_thz'

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, file), 'wb') as f:
            pickle.dump(self.thz_calib_data, f)
            pickle.dump(self.thz_slope, f)
            pickle.dump(self.thz_intersect, f)
            pickle.dump(self.thz_motor_name, f)
            log.info("Save THz calibration to a file: %s", os.path.join(path, file))

    def load_thz_calib(self, filepath):
        """Load THz pulse calibration from a file.
        """
        with open(filepath, 'rb') as f:
            self.thz_calib_data = pickle.load(f)
            self.thz_slope = pickle.load(f)
            self.thz_intersect = pickle.load(f)
            self.thz_motor_name = pickle.load(f)
            log.info("Load etof calibration from a file: %s", filepath)

    def process_hdf5_file(self, filepath, debug=False):
        """Load data for all registered spectrometers from an hdf5 file. This method is to be
        changed in order to adapt to a format of PALM data files in the future.

        Args:
            filepath: file path to be loaded

        Returns:
            tuple of tags and the corresponding results in a dictionary
        """
        data_raw = {}
        for etof_key in self.etofs:
            tags, data = get_tags_and_data(filepath, self.channels[etof_key])
            data_raw[etof_key] = data
            # data_raw[etof_key] = np.expand_dims(data[1, :], axis=0)

        results = self.process(data_raw, debug=debug)
        return (tags, *results)

    def _cross_corr_analysis(self, input_data, debug=False, peak='max'):
        """Perform analysis to determine arrival times via cross correlation.

        Usually, this data can be used to initally identify pulses that are falling within linear
        slope of a THz streak pulse.

        Args:
            input_data: input data to be correlated
            debug: (optional) return debug data

        Returns:
            pulse arrival delays via cross-correlation method
        """
        data_str = input_data['1']
        data_ref = input_data['0']

        corr_results = np.empty_like(data_ref)
        for i, (x, y) in enumerate(zip(data_ref, data_str)):
            corr_results[i, :] = np.correlate(x, y, mode='same')

        corr_res_uncut = corr_results.copy()
        corr_results = self._truncate_highest_peak(corr_results, 0)

        lags = self.energy_range - self.energy_range[int(self.energy_range.size / 2)]

        if peak == 'com':
            delays, _ = self._peak_params(lags, corr_results)
        elif peak == 'max':
            delays = lags[np.argmax(corr_results, axis=1)]

        pulse_lengths = self._peak_center_of_mass(input_data, lags)

        if debug:
            return delays, pulse_lengths, (input_data, lags, corr_res_uncut, corr_results)
        return delays, pulse_lengths

    def _deconvolution_analysis(self, input_data, iterations=200, debug=False):
        """Perform analysis to determine temporal profile of photon pulses.

        Args:
            input_data: data to be analysed
            iterations: (optional) number of iterations for the deconvolution analysis
            debug: (optional) return debug data

        Returns:
            result(s) of deconvolution
        """
        data_str = input_data['1']
        data_ref = input_data['0']

        deconv_result = np.empty_like(data_str)
        for i, (x, y) in enumerate(zip(data_ref, data_str)):
            deconv_result[i] = richardson_lucy_deconv(x, y, iterations=iterations)

        if debug:
            return deconv_result, input_data
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
        peak_var = np.sum((x - peak_mean[:, np.newaxis]) ** 2 * y, axis=1) / denom

        return peak_mean, peak_var

    @staticmethod
    def _truncate_highest_peak(y, thr):
        """Truncate the highest peak above a specified threshold.

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
                ind_max = np.argmax(y_1d)
                ind_max_height = np.argmax((ind_out > ind_max) & (ind_max > ind_in))

                y_1d[: ind_in[ind_max_height]] = 0
                y_1d[ind_out[ind_max_height] :] = 0

            return y_1d

        if y.ndim == 1:
            test_fun(y)
        else:
            np.apply_along_axis(test_fun, 1, y)

        return y

    @staticmethod
    def _truncate_widest_peak(y, thr):
        """Truncate the widest peak above a specified threshold.

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

                y_1d[: ind_in[ind_max_length]] = 0
                y_1d[ind_out[ind_max_length] :] = 0

            return y_1d

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
        data_ref = input_data['0'].copy()

        # thr1 = np.mean(self.spectrometers['1'].noise_std)
        # thr3 = np.mean(self.spectrometers['0'].noise_std)

        data_str = self._truncate_highest_peak(data_str, 0)
        data_ref = self._truncate_highest_peak(data_ref, 0)

        _, var1 = self._peak_params(lags, data_str)
        _, var3 = self._peak_params(lags, data_ref)

        ind = np.logical_and(~np.isnan(var1), ~np.isnan(var3))

        pulse_length = np.real(np.lib.scimath.sqrt(var1[ind] - var3[ind]))

        return pulse_length


def get_energy_from_filename(filename):
    """Parse filename and return energy value (first float number encountered). This method is
    likely to be changed in order to adapt to a format of PALM callibration files in the future.

    Args:
        filename: filename to be parsed

    Returns:
        energy in eV as a floating point number
    """
    energy = float(re.findall(r'\d+', filename)[0])

    return energy


def get_tags_and_data(filepath, etof_path):
    """Read PALM waveforms from an hdf5 file.

    Args:
        filepath: path to an hdf5 file
        etof_path: location of data in hdf5 file

    Returns:
        tags and data
    """
    # TODO: for the E1130 pylint issue, see
    # https://github.com/PyCQA/pylint/issues/2436
    with h5py.File(filepath, 'r') as h5f:
        try:
            tags = h5f['/pulseId'][:]
            data = -h5f['/{}'.format(etof_path)][:]  # pylint: disable=E1130
            return tags, data
        except (KeyError, AttributeError):
            pass

        try:
            tags = h5f['/scan 1/SLAAR21-LMOT-M552:MOT.VAL'][:]
            data = -h5f['/scan 1/{} averager'.format(etof_path)][:]  # pylint: disable=E1130
            return tags, data
        except (KeyError, AttributeError):
            pass

        try:
            tags = h5f['/data/{}/pulse_id'.format(etof_path)][:]
            data = -h5f['/data/{}/data'.format(etof_path)][:]  # pylint: disable=E1130
            return tags, data
        except (KeyError, AttributeError):
            pass

        try:
            tags = h5f['/pulse_id'][:]
            data = -h5f['/{}/data'.format(etof_path)][:]  # pylint: disable=E1130
            return tags, data
        except (KeyError, AttributeError):
            pass

        try:
            tags = []
            data = -h5f['/{}'.format(etof_path)][:]  # pylint: disable=E1130
            return tags, data
        except (KeyError, AttributeError):
            pass

        raise Exception("Could not locate data in {}".format(filepath))


def richardson_lucy_deconv(streaked_signal, reference_signal, iterations=200, noise=0.3):
    """Deconvolve eTOF waveforms using Richardson-Lucy algorithm, extracting pulse profile in
    a time domain.

    The assumption is that the streaked waveform was created by convolving an with a point-spread
    function PSF and possibly by adding noise.

    Args:
        streaked_signal: waveform after streaking
        reference_signal: waveform without effect of streaking
        iterations: number of Richardson-Lucy algorithm iterations
        noise: noise level in the units of waveform intensity

    Returns:
        pulse profile in a time domain
    """
    from numpy import conjugate, real, ones
    from numpy.fft import fft, ifft, fftshift

    weight = ones(streaked_signal.shape)

    otf = fft(fftshift(reference_signal))  # optical transfer function
    time_profile = streaked_signal.copy()
    time_profile.clip(min=0)

    weighted_signal = streaked_signal.copy() + noise
    weighted_signal = weight * weighted_signal.clip(min=0)

    scale = real(ifft(conjugate(otf) * fft(weight)))

    for _ in range(iterations):
        relative_psf = weighted_signal / (real(ifft(otf * fft(time_profile))) + noise)
        time_profile *= real(ifft(conjugate(otf) * fft(relative_psf))) / scale
        time_profile.clip(min=0)

    return time_profile
