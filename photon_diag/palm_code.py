import os
import re

import h5py
import numpy as np

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

        self.spectrometers = {'1': Spectrometer(chan='SAROP11-PALMK118:CH2_BUFFER'),
                              '0': Spectrometer(chan='SAROP11-PALMK118:CH1_BUFFER')}

        self.hdf5_range = [0, 2000]
        self.tags = []
        self.interp_energy = np.linspace(1, 120, 500)

    def __call__(self, waveforms, method='xcorr', jacobian=False, noise_thr=3):
        """Main function to analyse PALM data that pipelines separate stages of data processing.

        Args:
            waveforms: dictionary with waveforms from streaked and non-streaked spectrometers
            method: (optional) currently, only one method is available {'xcorr' (default)}
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
                prep_data[etof_key] = etof(data, self.interp_energy, jacobian=jacobian, noise_thr=noise_thr)

            results = self._cross_corr_analysis(prep_data)

        else:
            raise RuntimeError(f"Method '{method}' is not recognised")

        return results, prep_data

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
            calibrated_etofs = etofs

        if overwrite:
            for etof in calibrated_etofs:
                etof.calib_data.drop(etof.calib_data.index[:], inplace=True)

        with os.scandir(calib_folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    energy = self._get_energy_from_filename(entry.name)

                    for etof in calibrated_etofs:
                        if not overwrite and energy in etof.calib_data.index:
                            continue

                        etof.internal_time = self._get_internal_time(*self.hdf5_range)
                        _, calib_waveforms = self._get_tags_and_data(entry.path, etof.chan, *self.hdf5_range)

                        # Filter out bad shots
                        calib_waveforms = calib_waveforms[(calib_waveforms > -5000).all(axis=1)]

                        etof.add_calibration_point(energy, calib_waveforms)

        calib_results = {}
        for etof_key in self.spectrometers.keys():
            calib_results[etof_key] = self.spectrometers[etof_key].fit_calibration_curve(bkg_en=bkg_en)

        return calib_results

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
            self.tags, data = self._get_tags_and_data(filepath, etof.chan, *self.hdf5_range)
            data_raw[etof_key] = data
            # data_raw[etof_key] = np.expand_dims(data[1, :], axis=0)
            time_raw[etof_key] = self._get_internal_time(*self.hdf5_range)

        # Filter out bad shots
        good_ind = (data_raw['0'] > -5000).all(axis=1) & (data_raw['1'] > -5000).all(axis=1)
        data_raw['0'] = data_raw['0'][good_ind, :]
        data_raw['1'] = data_raw['1'][good_ind, :]
        self.tags = self.tags[good_ind]

        results, prep_data = self(data_raw)

        return results, prep_data

    @staticmethod
    def _get_internal_time(first_ind=None, last_ind=None):
        """Get PALM internal time in spectrometer readout units. Current setup produces 2000 points (for a
        span of 400 ns).

        Args:
            first_ind: (optional) index of a first element to read
            last_ind: (optional) index of a last element to read

        Returns:
            time: internal electron time-of-flight reference.
        """

        return np.arange(0, 2000)[first_ind:last_ind]

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
        with h5py.File(filepath, 'r') as h5f:
            tags = h5f['/pulse_id'][:]
            data = -h5f[f'/{etof_path}/data'][:, first_ind:last_ind]

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
        energy = float(re.findall('\d+', filename)[0])

        return 1510 - energy

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

        corr_results = self._truncate_highest_peak(corr_results, 0)

        lags = self.interp_energy - self.interp_energy[int(self.interp_energy.size/2)]

        delays, _ = self._peak_params(lags, corr_results)

        pulse_lengths = self._peak_center_of_mass(input_data, lags)

        return lags, delays, pulse_lengths

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

        # thr1 = np.mean(self.spectrometers['1'].noise_std)
        # thr3 = np.mean(self.spectrometers['0'].noise_std)

        data_str = self._truncate_highest_peak(data_str, 0)
        data_nonstr = self._truncate_highest_peak(data_nonstr, 0)

        _, var1 = self._peak_params(lags, data_str)
        _, var3 = self._peak_params(lags, data_nonstr)

        ind = np.logical_and(~np.isnan(var1), ~np.isnan(var3))

        pulse_length = np.real(np.lib.scimath.sqrt(var1[ind] - var3[ind]))

        return pulse_length
