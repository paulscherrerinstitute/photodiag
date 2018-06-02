import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Spectrometer:
    """Class describing a single eTOF spectrometer.
    """
    def __init__(self, chan, a=None, b=None):
        """ Initialize Spectrometer object.

        Args:
            chan: channel name of the spectrometer data
            a: calibration constant 'a'
            b: calibration constant 'b'
        """
        self.chan = chan
        self.calib_a = a
        self.calib_b = b

        # index of self.calib_data DataFrame is 'energy'
        self.calib_data = pd.DataFrame(data={'waveform': np.array([], dtype=float),
                                             't0': np.array([], dtype=int),
                                             'noise_mean': np.array([], dtype=float),
                                             'noise_std': np.array([], dtype=float)})

        # current setup outputs 2000 points for a span of 400 ns excluding the end point
        self.internal_time = np.linspace(0, 400, 2000, endpoint=False)
        self.noise_range = [1900, 2000]
        self.data_range = [300, 1000]
        self.t0 = np.empty(0)

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

        t0, _ampl = self._detect_photon_peak(data_avg, noise_std)

        self.calib_data.loc[energy] = {'waveform': data_avg,
                                       't0': t0,
                                       'noise_mean': noise_mean,
                                       'noise_std': noise_std}

    def fit_calibration_curve(self, bkg_en=None):
        """Perform fitting of calibration data.

        Args:
            bkg_en: (optional) subtract background energy profile of that value (eV)

        Returns:
            calibration constants and a goodness of fit
        """
        cd = self.calib_data
        calib_t0 = cd.t0
        calib_wf = cd.waveform

        if bkg_en is not None:
            if bkg_en not in cd.index:
                raise Exception('Can not find data for background energy')

            calib_t0 = calib_t0.loc[cd.index != bkg_en]
            calib_wf = calib_wf.loc[cd.index != bkg_en] - calib_wf.loc[bkg_en]

        calib_peak_pos = self.data_range[0] + calib_wf.apply(lambda x: x[slice(*self.data_range)].argmax())

        time_delays = self.internal_time[calib_peak_pos] - self.internal_time[calib_t0]
        pulse_energies = calib_wf.index

        def fit_func(time, a, b):
            return (a / time) ** 2 + b

        popt, _pcov = curve_fit(fit_func, time_delays, pulse_energies)

        self.calib_a, self.calib_b = popt
        self.t0 = np.round(cd.t0.mean()).astype(int)

        return popt, time_delays, pulse_energies

    def convert(self, input_data, interp_energy, jacobian=False, noise_thr=3):
        """Perform electron time of flight (eTOF) to pulse energy transformation (ns -> eV) of data through
        the spectrometer's calibration constants and a photon peak position followed by 1D interpolation.

        Args:
            input_data: data to be processed
            jacobian: apply jacobian corrections of spectrometer's time to energy transformation
            noise_thr:

        Returns:
            interpolated output data
        """
        flight_time = self.internal_time[self.t0 + 1:] - self.internal_time[self.t0]
        pulse_energy = (self.calib_a / flight_time) ** 2 + self.calib_b

        output_data = input_data[:, self.t0 + 1:]

        if jacobian:
            jacobian_factor_inv = -pulse_energy ** (3 / 2)  # = 1 / jacobian_factor
            output_data = output_data / jacobian_factor_inv  # = spectr.data * jacobian_factor

        def interpolate_row(data, energy, interp_energy):
            return np.interp(interp_energy, energy, data)

        output_data = np.apply_along_axis(interpolate_row, 1,
                                          output_data[:, ::-1], pulse_energy[::-1], interp_energy)

        output_data = output_data - noise_thr * self.calib_data.noise_std.mean()

        return output_data

    @staticmethod
    def _detect_photon_peak(waveform, noise_std, noise_thr=3):
        """Estimate position and amplitude of a photon peak.

        Under assumption that the photon peak is the first peak encontered above the specified noise
        level (= noise_thr * noise_std).

        Args:
            waveform: waveform of interest
            noise_std: noise level in waveform units
            noise_thr: number of noise_std standard deviations above which the signal is considered to be
                detectable (default is 3-sigma)

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
