import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Spectrometer:
    """Class describing a single eTOF spectrometer.
    """
    def __init__(self, noise_range):
        """ Initialize Spectrometer object.

        Args:
            noise_range: a range of spectrometer bins that does not contain any signals
        """
        # index of self.calib_data DataFrame is 'energy'
        self.calib_data = pd.DataFrame({
            'waveform': np.array([], dtype=float),
            'calib_t0': np.array([], dtype=float),
            'calib_tpeak': np.array([], dtype=float),
            'noise_mean': np.array([], dtype=float),
            'noise_std': np.array([], dtype=float)})

        self.calib_a = None
        self.calib_b = None
        self.internal_time_bins = None
        self.noise_range = noise_range
        self.calib_t0 = np.empty(0)

    def add_calibration_point(self, energy, calib_waveforms):
        """Add calibration point for a specified X-ray energy.

        Args:
            energy: X-ray energy used in data acquisition
            calib_waveforms: calibration data as a 2D array
        """
        if self.internal_time_bins:
            if self.internal_time_bins != calib_waveforms.shape[1]:
                raise Exception('eTOF number of bins is inconsistent.')
        else:
            self.internal_time_bins = calib_waveforms.shape[1]

        noise = calib_waveforms[:, slice(*self.noise_range)]
        noise_mean = noise.mean(axis=1).mean()
        noise_std = noise.std(axis=1).mean()

        # keep axis=0 for a situation with a single waveform
        waveform = calib_waveforms.mean(axis=0) - noise_mean

        calib_t0, _ = self._detect_photon_peak(waveform, noise_std)
        try:
            calib_tpeak = self._detect_electron_peak(waveform, noise_std)
        except:
            calib_tpeak = np.nan

        self.calib_data.loc[energy] = {
            'waveform': waveform,
            'calib_t0': calib_t0,
            'calib_tpeak': calib_tpeak,
            'noise_mean': noise_mean,
            'noise_std': noise_std}

    def fit_calibration_curve(self):
        """Perform fitting of calibration data.

        Returns:
            calibration constants and a goodness of fit
        """
        self.calib_t0 = np.round(self.calib_data['calib_t0'].median()).astype(int)
        time_delays_df = self.calib_data['calib_tpeak'] - self.calib_t0

        # convert to numpy arrays
        time_delays = time_delays_df.values
        pulse_energies = time_delays_df.index.values

        def fit_func(time, a, b):
            return (a / time) ** 2 + b

        valid = ~(np.isnan(time_delays) | np.isnan(pulse_energies))
        popt, _pcov = curve_fit(fit_func, time_delays[valid], pulse_energies[valid])
        self.calib_a, self.calib_b = popt

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
            jacobian_factor_inv = - pulse_energy ** (3 / 2)  # = 1 / jacobian_factor
            output_data /= jacobian_factor_inv  # = spectr.data * jacobian_factor

        def interpolate_row(data, energy, interp_energy):
            return np.interp(interp_energy, energy, data)

        output_data = np.apply_along_axis(
            interpolate_row, 1, output_data[:, ::-1], pulse_energy[::-1], interp_energy)

        output_data -= noise_thr * self.calib_data['noise_std'].mean()

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
                to be detectable (default is 1-sigma)

        Returns:
            index of a photon peak maximum
            photon peak amplitude
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
        """Estimate position of an electron peak.

        Under assumption that the electron peak is the last peak encontered above the specified
        noise level (= noise_thr * noise_std).

        Args:
            waveform: waveform of interest
            noise_std: noise level in waveform units
            noise_thr: number of noise_std standard deviations above which the signal is considered
                to be detectable (default is 3-sigma)

        Returns:
            index of an electron peak maximum
        """
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
