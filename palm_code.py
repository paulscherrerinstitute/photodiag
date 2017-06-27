import numpy as np
# TODO: adapt to utilize pandas DataFrames(?)
# import pandas as pd


class PalmSetup(object):
    """Class describing the photon arrival and length monitor (PALM) setup
    """
    def __init__(self, root_path=None):
        # TODO: hardcoded values need to be interfaced with other modules or be a user specified
        self.root_path = root_path
        self.spectrometers = {'C1': Spectrometer(streak=1, a=787, c=8512),
                              'C2': Spectrometer(streak=-1, a=833.4, c=8606),
                              'C3': Spectrometer(streak=0, a=780.5, c=8516)}

        self.hdf5_data_range = [2000, 4000]
        self.proc_data_range = [8600, 9400]
        self.cross_corr_res = []
        self._tags = []
        self.psen = PsenSetup(root_path=root_path, ttcalib=2.6270)

    def load_palm_data(self, filename):
        self._read_hdf5_data(filename, tags=None,
                             first_ind=self.hdf5_data_range[0], last_ind=self.hdf5_data_range[1])

    # TODO: after clarification of hdf5 file structure and data format, refactor to a separate function
    def _read_hdf5_data(self, filename, tags, first_ind=2000, last_ind=4000):
        import h5py
        with h5py.File(self.root_path + filename) as f:
            self._tags = f['/tags'][:]
            # TODO: can eTOFs have different set of tags? -> make tags as Spectrometer attribute
            for spectr_name, spectr in self.spectrometers.items():
                spectr.time_raw = f['/{}/time'.format(spectr_name)][first_ind:last_ind] * 1e9  # convert to fs
                spectr.data_raw = -f['/{}/data'.format(spectr_name)][:, first_ind:last_ind]  # flip a signal

    def analyse_palm_data(self, stages='all', tags=None):
        """Analyse palm data.

        Args:
            stages: 'all'|'prepare'|..
            tags: apply analysis only to a specific set of tags

        Returns:

        """
        if stages == 'all':
            self._transform_palm_data()
            self._interpolate_palm_data()
            self._cross_corr_analysis()
            # self.deconvolve_analysis(tags=tags)

        elif stages == 'prepare':  # TODO: could be a good idea to perform only certain steps in a whole chain
            self._transform_palm_data()
            self._interpolate_palm_data()

    def _transform_palm_data(self):
        """Perform electron time of flight (eTOF) to pulse energy transformation (ns -> eV) through the
        spectrometer's calibration constants and a photon peak position.

        ## spectrometers: Dictionary, containing registered eTOF spectrometers

        Returns:
        """
        for spectr in self.spectrometers.values():
            spectr.time = spectr.time_raw[spectr.t0 + 1:] - spectr.time_raw[spectr.t0]
            spectr.energy = spectr.calib_c + (spectr.calib_a / spectr.time) ** 2
            spectr.data = spectr.data_raw[:, spectr.t0 + 1:]

    def _interpolate_palm_data(self):
        """Perform data 1D interpolation.

        ## spectrometers: Dictionary, containing registered eTOF spectrometers

        Returns:
        """
        def interpolate_row(data, energy, interp_energy):
            return np.interp(interp_energy, energy, data)

        for spectr in self.spectrometers.values():
            spectr.interp_data = np.apply_along_axis(interpolate_row, 1, spectr.data[:, ::-1],
                                                     spectr.energy[::-1], spectr.interp_energy)

    def _cross_corr_analysis(self, tags=None):
        """Perform analysis to determine arrival times via cross correlation.

        Usually, this data can be used to initally identify pulses that are falling within linear slope of
        a THz streak pulse.

        Args:
            tags:

        Returns:
            ## result(s) of cross-correlation
        """
        if tags is None:  # process all tags
            spectr_nonstreaked = self.spectrometers['C3']
            spectr_streaked = self.spectrometers['C1']
            corr_result = np.empty_like(spectr_nonstreaked.interp_data)
            for i, (x, y) in enumerate(zip(spectr_nonstreaked.interp_data, spectr_streaked.interp_data)):
                corr_result[i] = np.correlate(x, y, mode='same')

        else:
            x = self.spectrometers['C3'].interp_data[tags, :]
            y = self.spectrometers['C1'].interp_data[tags, :]
            corr_result = np.correlate(x, y, mode='same')

        self.cross_corr_res = corr_result

    def _deconvolve_analysis(self, iterations=200, tags=None):
        """Perform analysis to determine temporal profile of photon pulses.

        Args:
            iterations: number of iterations for the deconvolution analysis
            tags: apply analysis only to a specific set of tags

        Returns:
            ## result(s) of deconvolution
        """
        if tags is None:  # process all tags
            spectr1 = self.spectrometers['C3']
            spectr2 = self.spectrometers['C1']
            deconv_result = np.empty_like(spectr1.interp_data)
            for i, (x, y) in enumerate(zip(spectr1.interp_data, spectr2.interp_data)):
                deconv_result[i] = richardson_lucy_deconv(x, y, iterations=iterations)

        else:
            x = self.spectrometers['C3'].interp_data[tags, :]
            y = self.spectrometers['C1'].interp_data[tags, :]
            deconv_result = richardson_lucy_deconv(y, x, iterations=iterations)

        return deconv_result

    # TODO: some functions planned to extend the public API
    def add_spectrometer(self, name=None, streak=0):
        if name is None:
            pass
        else:
            self.spectrometers[name] = Spectrometer(streak)


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
        ## pulse profile in a time domain
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


class Spectrometer(object):
    """Class describing a single eTOF spectrometer.
    """

    def __init__(self, streak=0, a=None, b=None, c=None):
        """ Initialize Spectrometer object.

        Args:
            streak: presense of a streaking field (0: no streaking, 1: positive streaking, -1: negative
            streaking)
            a: calibration constant 'a'
            b: calibration constant 'b' (currently unused)
            c: calibration constant 'c'
        """
        self.streak = streak
        self.calib_a = a
        self.calib_b = b
        self.calib_c = c

        self.data_raw = np.empty(0)
        self.time_raw = np.empty(0)
        self.data = np.empty(0)
        self.time = np.empty(0)
        self.interp_data = np.empty(0)
        self.interp_energy = np.arange(8800, 9312)
        self.noise_range = [0, 500]
        self.noise_mean = 0
        self.noise_std = 0
        self.t0 = 0
        self.energy = np.empty(0)

    def detect_photon_peak(self, noise_thr=3):
        """Estimate position of the photon peak.

        Under assumption that the photon peak is the first peak encontered above the specified noise
        threshold level.

        Args:
            noise_thr: Number of noise standard deviations above which the signal is considered to be
                detectable (default is 3-sigma).

        Returns:
            Index of the photon peak position.
        """
        data = self.data_raw.mean()
        data_above_thr = np.greater(data, noise_thr*self.noise_std)

        ind_l = np.argmax(data_above_thr)
        ind_r = ind_l + np.argmin(data_above_thr[ind_l:])

        return ind_l + np.argmax(data[ind_l:ind_r])

    def noise_params(self):
        """Calculate noise parameters.

        Returns:
            Mean and standard deviation of noise fluctuations for each shot.
        """
        noise = self.data_raw[:, slice(*self.noise_range)]
        mean = noise.mean(axis=1)
        std = noise.std(axis=1)

        return mean, std


# TODO: currently, this class works with SACLA timing tool data -> generalize to SwissFEL PSEN
class PsenSetup(object):
    """Class describing photon spectral encoder (PSEN) setup.
    """
    def __init__(self, root_path, ttcalib):
        """Initialize PsenSetup object

        Args:
            root_path: path to the folder with data files
            ttcalib: calibration constant of the timing tool in [fs/pixel]
        """
        self.root_path = root_path
        self.ttcalib = ttcalib
        self.tags = []
        self.data = []

    def load_psen_data(self, filename, method='average', remove_nans=True):
        """Load data for the photon spectral encoder (PSEN), currently works with SACLA timing tool data.

        Args:
            filename: name of a csv file with data
            method: {'derivative', 'fitting', 'average'(default)} available methods of timing edge detection
            remove_nans: clear data from NaNs

        Returns:

        """
        data = np.genfromtxt(self.root_path + filename, delimiter=',', skip_header=2)
        self.tags = data[:, 0]  # first column is a tags list

        if method == 'derivative':
            data = data[:, 1]  # second column are timing edge derivative values (pixel)
        elif method == 'fitting':
            data = data[:, 2]  # first column are timing edge fitting values (pixel)
        elif method == 'average':
            data = (data[:, 1]+data[:, 2])/2
        else:
            raise RuntimeError("Method '{}' is not recognised".format(method))

        self.data = data*self.ttcalib  # convert to fs

        if remove_nans:
            idx = ~np.isnan(data)
            self.tags = self.tags[idx]
            self.data = self.data[idx]
