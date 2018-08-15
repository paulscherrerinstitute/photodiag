import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Spectrometer:
    """Class describing a single eTOF spectrometer.
    """
    def __init__(self):
        """ Initialize Spectrometer object.
        """
        # index of self.calib_data DataFrame is 'energy'
        self.calib_data = pd.DataFrame({
            'waveform': np.array([], dtype=float),
            'calib_t0': np.array([], dtype=int),
            'calib_tpeak': np.array([], dtype=int),
            'noise_mean': np.array([], dtype=float),
            'noise_std': np.array([], dtype=float)})

        self.calib_a = None
        self.calib_b = None
        self.internal_time_bins = 2000
        self.noise_range = [0, 250]
        self.calib_t0 = np.empty(0)

    def add_calibration_point(self, energy, calib_waveforms):
        """Add calibration point for a specified X-ray energy.

        Args:
            energy: X-ray energy used in data acquisition
            calib_waveforms: calibration data as a 2D array
        """
        noise = calib_waveforms[:, slice(*self.noise_range)]
        noise_mean = noise.mean(axis=1).mean()
        noise_std = noise.std(axis=1).mean()

        data_avg = calib_waveforms.mean(axis=0)
        data_avg = data_avg - data_avg[slice(*self.noise_range)].mean()

        calib_t0, _ampl = self._detect_photon_peak(data_avg, noise_std)
        calib_tpeak = self._detect_electron_peak(data_avg, noise_std)

        self.calib_data.loc[energy] = {
            'waveform': data_avg,
            'calib_t0': calib_t0,
            'calib_tpeak': calib_tpeak,
            'noise_mean': noise_mean,
            'noise_std': noise_std}

    def fit_calibration_curve(self):
        """Perform fitting of calibration data.

        Returns:
            calibration constants and a goodness of fit
        """
        cd = self.calib_data
        calib_t0 = cd.calib_t0
        calib_wf = cd.waveform
        calib_tpeak = cd.calib_tpeak

        time_delays = calib_tpeak - calib_t0
        pulse_energies = calib_wf.index

        def fit_func(time, a, b):
            return (a / time) ** 2 + b

        popt, _pcov = curve_fit(fit_func, time_delays, pulse_energies)

        self.calib_a, self.calib_b = popt
        self.calib_t0 = np.round(cd.calib_t0.mean()).astype(int)

        return popt, time_delays, pulse_energies

    def convert(self, input_data, interp_energy, jacobian=False, noise_thr=3):
        """Perform electron time of flight (eTOF) to pulse energy transformation (ns -> eV) of data
        through the spectrometer's calibration constants and a photon peak position followed by
        1D interpolation.

        Args:
            input_data: data to be processed
            jacobian: apply jacobian corrections of spectrometer's time to energy transformation
            noise_thr:

        Returns:
            interpolated output data
        """
        flight_time = np.arange(1, self.internal_time_bins - self.calib_t0)
        pulse_energy = (self.calib_a / flight_time) ** 2 + self.calib_b

        output_data = input_data[:, self.calib_t0 + 1:]

        if jacobian:
            jacobian_factor_inv = -pulse_energy ** (3 / 2)  # = 1 / jacobian_factor
            output_data = output_data / jacobian_factor_inv  # = spectr.data * jacobian_factor

        def interpolate_row(data, energy, interp_energy):
            return np.interp(interp_energy, energy, data)

        output_data = np.apply_along_axis(
            interpolate_row, 1, output_data[:, ::-1], pulse_energy[::-1], interp_energy)

        output_data = output_data - noise_thr * self.calib_data.noise_std.mean()

        return output_data

    @staticmethod
    def _detect_photon_peak(waveform, noise_std, noise_thr=1):
        """Estimate position and amplitude of a photon peak.

        Under assumption that the photon peak is the first peak encontered above the specified
        noise level (= noise_thr * noise_std).

        Args:
            waveform: waveform of interest
            noise_std: noise level in waveform units
            noise_thr: number of noise_std standard deviations above which the signal is considered
                to be detectable (default is 3-sigma)

        Returns:
            index of the photon peak position
        """
        above_thr = np.greater(waveform, noise_thr * noise_std)

        # TODO: the code could be improved once the following issue is resolved,
        # https://github.com/numpy/numpy/issues/2269
        if not above_thr.any():
            # no values above the noise threshold in the waveform
            raise Exception('Can not detect a photon peak')

        ind_l = np.argmax(above_thr)

        if not above_thr[ind_l:].any():
            # no values below the noise threshold along the peak
            raise Exception('Can not detect a photon peak')

        ind_r = ind_l + np.argmin(above_thr[ind_l:])

        position = ind_l + np.argmax(waveform[ind_l:ind_r])
        amplitude = waveform[position]

        return position, amplitude

    @staticmethod
    def _detect_electron_peak(waveform, noise_std, noise_thr=3):
        above_thr = np.greater(waveform[::-1], noise_thr * noise_std)

        # TODO: the code could be improved once the following issue is resolved,
        # https://github.com/numpy/numpy/issues/2269
        if not above_thr.any():
            # no values above the noise threshold in the waveform
            raise Exception('Can not detect a peak')

        ind_l = np.argmax(above_thr)

        if not above_thr[ind_l:].any():
            # no values below the noise threshold along the peak
            raise Exception('Can not detect a peak')

        ind_r = ind_l + np.argmin(above_thr[ind_l:])
        ind_l = len(above_thr) - ind_l - 1
        ind_r = len(above_thr) - ind_r - 1

        position = ind_r + np.argmax(waveform[ind_r:ind_l])

        return position
