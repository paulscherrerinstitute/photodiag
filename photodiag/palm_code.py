import datetime
import os
import pickle
import re

import h5py
import numpy as np

from photodiag.spectrometer import Spectrometer


class PalmSetup:
    """Class describing the photon arrival and length monitor (PALM) setup.
    """
    def __init__(self, unstr_chan, str_chan):
        """Initialize PALM setup object.

        For the electron time of flight (eTOF) spectrometers the following notation is used: presense of a
        streaking field ('0': no streaking, '1': positive streaking)
        """
        self.etofs = {'0': Spectrometer(chan=unstr_chan), '1': Spectrometer(chan=str_chan)}
        self.interp_energy = np.linspace(2850, 3100, 300)

    def calibrate_etof(self, folder_name, bkg_en=None, etofs=None, overwrite=True):
        """General routine for a calibration process of the eTOF spectrometers.

        Args:
            folder_name: location of hdf5 files with calibration data
            bkg_en: (optional) background energy profile to be subtracted from other waveforms (e.g. to
                    remove influence of Auger peaks)
            etofs: (optional) list of eTOF spectrometers to be calibrated
            overwrite: (optional) start over a calibration process

        Returns:
            results of calibration as dictionary
        """
        if etofs is None:
            calibrated_etofs = self.etofs.values()
        else:
            calibrated_etofs = etofs

        if overwrite:
            for etof in calibrated_etofs:
                etof.calib_data.drop(etof.calib_data.index[:], inplace=True)

        with os.scandir(folder_name) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    energy = self._get_energy_from_filename(entry.name)

                    for etof in calibrated_etofs:
                        if not overwrite and energy in etof.calib_data.index:
                            continue

                        _, calib_waveforms = self._get_tags_and_data(entry.path, etof.chan)

                        etof.add_calibration_point(energy, calib_waveforms)

        calib_results = {}
        for etof_key in self.etofs:
            calib_results[etof_key] = self.etofs[etof_key].fit_calibration_curve(bkg_en=bkg_en)

        return calib_results

    def save_etof_calib(self, file=None):
        """ Save eTOF calibration to a file.
        """
        if not file:
            file = f"eTOF_calib_{datetime.datetime.now().isoformat(sep='_', timespec='seconds')}"

        path, file = os.path.split(file)

        if not path:
            path = os.path.join(os.path.expanduser('~'), 'eTOF_calibs')

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, file), 'wb') as f:
            pickle.dump(self.etofs, f)

    def load_etof_calib(self, file):
        """Load eTOF calibration from a file.
        """
        with open(file, 'rb') as f:
            self.etofs = pickle.load(f)

    def process(self, waveforms, method='xcorr', jacobian=False, noise_thr=3, debug=False):
        """Main function to analyse PALM data that pipelines separate stages of data processing.

        Args:
            waveforms: dictionary with waveforms from streaked and non-streaked spectrometers
            method: (optional) currently, only one method is available {'xcorr' (default), 'deconv'}
            jacobian: (optional) apply jacobian corrections of spectrometer's time to energy transformation
            noise_thr:
            debug: (optional) return debug data

        Returns:
            pulse lengths and arrival times per pulse
        """
        prep_data = {}
        for etof_key, data in waveforms.items():
            etof = self.etofs[etof_key]
            prep_data[etof_key] = etof.convert(
                data, self.interp_energy, jacobian=jacobian, noise_thr=noise_thr)

        if method == 'xcorr':
            results = self._cross_corr_analysis(prep_data, debug=debug)

        elif method == 'deconv':
            results = self._deconvolution_analysis(prep_data, debug=debug)

        else:
            raise RuntimeError(f"Method '{method}' is not recognised")

        return results

    def calibrate_thz(self):
        """Calibrate THz pulse.
        """
        pass

    def save_thz_calib(self, file):
        """ Save THz pulse calibration to a file.
        """
        pass

    def load_thz_calib(self, file):
        """Load THz pulse calibration from a file.
        """
        pass

    def process_hdf5_file(self, filepath, debug=False):
        """Load data for all registered spectrometers from an hdf5 file. This method is to be changed
        in order to adapt to a format of PALM data files in the future.

        Args:
            filepath: file path to be loaded

        Returns:
            tuple of tags and the corresponding results in a dictionary
        """
        data_raw = {}
        for etof_key, etof in self.etofs.items():
            tags, data = self._get_tags_and_data(filepath, etof.chan)
            data_raw[etof_key] = data
            # data_raw[etof_key] = np.expand_dims(data[1, :], axis=0)

        results = self.process(data_raw, debug=debug)
        return (tags, *results)

    @staticmethod
    def _get_tags_and_data(filepath, etof_path):
        """Read PALM waveforms from an hdf5 file.

        Args:
            filepath: path to an hdf5 file
            etof_path: location of data in hdf5 file

        Returns:
            tags and data
        """
        with h5py.File(filepath, 'r') as h5f:
            if 'monofiles' in filepath:  # calibration data
                tags = h5f['/pulseId'][:]
                data = -h5f[f'/{etof_path}'][:]
            else:
                try:
                    tags = h5f['/scan 1/SLAAR21-LMOT-M552:MOT.VAL'][:]
                    data = -h5f[f'/scan 1/{etof_path} averager'][:]
                except KeyError:
                    tags = h5f[f'/data/{etof_path}/pulse_id'][:]
                    data = -h5f[f'/data/{etof_path}/data'][:]

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
        energy = float(re.findall(r'\d+', filename)[0])

        return energy

    def _cross_corr_analysis(self, input_data, debug=False):
        """Perform analysis to determine arrival times via cross correlation.

        Usually, this data can be used to initally identify pulses that are falling within linear slope of
        a THz streak pulse.

        Args:
            input_data: input data to be correlated
            debug: (optional) return debug data

        Returns:
            pulse arrival delays via cross-correlation method
        """
        data_str = input_data['1']
        data_nonstr = input_data['0']

        corr_results = np.empty_like(data_nonstr)
        for i, (x, y) in enumerate(zip(data_nonstr, data_str)):
            corr_results[i, :] = np.correlate(x, y, mode='same')

        corr_res_uncut = corr_results.copy()
        corr_results = self._truncate_highest_peak(corr_results, 0)

        lags = self.interp_energy - self.interp_energy[int(self.interp_energy.size/2)]

        delays, _ = self._peak_params(lags, corr_results)
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
        data_nonstr = input_data['0']

        deconv_result = np.empty_like(data_str)
        for i, (x, y) in enumerate(zip(data_nonstr, data_str)):
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
        peak_var = np.sum((x - peak_mean[:, np.newaxis])**2 * y, axis=1) / denom

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

                y_1d[:ind_in[ind_max_height]] = 0
                y_1d[ind_out[ind_max_height]:] = 0

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

        # thr1 = np.mean(self.spectrometers['1'].noise_std)
        # thr3 = np.mean(self.spectrometers['0'].noise_std)

        data_str = self._truncate_highest_peak(data_str, 0)
        data_nonstr = self._truncate_highest_peak(data_nonstr, 0)

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
