import numpy as np
from scipy.optimize import curve_fit


class Spectrometer:
    """Class describing a single eTOF spectrometer.
    """
    def __init__(self, path, a=None, b=None):
        """ Initialize Spectrometer object.

        Args:
            path: path to data in hdf5 file
            a: calibration constant 'a'
            b: calibration constant 'b'
        """
        self.path = path

        self.calib_a = a
        self.calib_b = b
        self.calib_data = {}

        self.internal_time = np.empty(0)
        self.time = np.empty(0)

        self.interp_energy = np.arange(8700, 9300)
        self.noise_range = [0, 2000]
        self.data_range = [3000, 4000]
        self.noise_mean = None
        self.noise_std = None
        self.t0 = np.empty(0)
        self.energy = np.empty(0)

    def __call__(self, input_data, noise_thr=7, jacobian=False):
        """Perform electron time of flight (eTOF) to pulse energy transformation (ns -> eV) of data through
        the spectrometer's calibration constants and a photon peak position followed by 1D interpolation.

        Args:
            input_data: data to be processed
            noise_thr:
            jacobian: apply jacobian corrections of spectrometer's time to energy transformation

        Returns:
            interpolated output data
        """
        self.time = self.internal_time[self.t0 + 1:] - self.internal_time[self.t0]
        self.energy = (self.calib_a / self.time) ** 2 + self.calib_b

        output_data = input_data[:, self.t0 + 1:]

        if jacobian:
            jacobian_factor_inv = -self.energy ** (3 / 2)  # = 1 / jacobian_factor
            output_data = output_data / jacobian_factor_inv  # = spectr.data * jacobian_factor

        def interpolate_row(data, energy, interp_energy):
            return np.interp(interp_energy, energy, data)

        output_data = np.apply_along_axis(interpolate_row, 1, output_data[:, ::-1],
                                          self.energy[::-1], self.interp_energy)

        output_data = output_data - noise_thr * np.mean(self.noise_std)

        return output_data

    def add_calibration_point(self, energy, calib_waveforms):
        """Add calibration point for a specified X-ray energy.

        Args:
            energy: calibration energy used in data acquisition
            calib_waveforms: calibration data as a 2D array
        """
        noise = calib_waveforms[:, slice(*self.noise_range)]
        self.noise_mean = noise.mean(axis=1)
        self.noise_std = noise.std(axis=1)

        data_avg = calib_waveforms.mean(axis=0)
        data_avg = data_avg - data_avg[slice(*self.noise_range)].mean()

        t0, ampl = self._detect_photon_peak(data_avg, self.noise_std.mean())
        data_avg = data_avg / ampl

        self.calib_data[energy] = (data_avg, t0)

    def fit_calibration_curve(self, bkg_en=None):
        """Perform fitting of calibration data.

        Args:
            bkg_en: (optional) subtract background energy profile of that value (eV)

        Returns:
            calibration constants and a goodness of fit
        """
        def fit_func(time, a, b):
            return (a / time) ** 2 + b

        if bkg_en not in self.calib_data.keys():
            raise Exception('Can not find background energy')

        calib_waveforms = []
        calib_t0 = []
        pulse_energies = []
        for en, (waveform, t0) in self.calib_data.items():
            calib_waveforms.append(waveform)
            calib_t0.append(t0)
            pulse_energies.append(en)

        if bkg_en is not None:
            bkg_ind = pulse_energies.index(bkg_en)
            bkg_waveform = np.array(calib_waveforms[bkg_ind])
            del calib_waveforms[bkg_ind]
            del calib_t0[bkg_ind]
            del pulse_energies[bkg_ind]
            calib_waveforms = np.array(calib_waveforms)
            calib_waveforms -= bkg_waveform

        calib_peak_pos = self.data_range[0] + np.argmax(calib_waveforms[:, slice(*self.data_range)], axis=1)
        time_delays = self.internal_time[calib_peak_pos] - self.internal_time[calib_t0]

        popt, pcov = curve_fit(fit_func, time_delays, pulse_energies)

        self.calib_a, self.calib_b = popt
        self.t0 = np.round(np.mean(calib_t0)).astype(int)

        return popt, time_delays, pulse_energies

    @staticmethod
    def _detect_photon_peak(waveform, noise_std, noise_thr=3):
        """Estimate position and amplitude of a photon peak.

        Under assumption that the photon peak is the first peak encontered above the specified noise_std
        threshold level.

        Args:
            waveform:
            noise_std: noise level in waveform units
            noise_thr: number of noise_std standard deviations above which the signal is considered to be
                detectable (default is 3-sigma)

        Returns:
            index of the photon peak position
        """
        above_thr = np.greater(waveform, noise_thr * noise_std)

        ind_l = np.argmax(above_thr)
        ind_r = ind_l + np.argmin(above_thr[ind_l:])

        position = ind_l + np.argmax(waveform[ind_l:ind_r])
        amplitude = waveform[position]

        return position, amplitude
