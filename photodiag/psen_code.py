"""
The class currently works with SACLA timing tool results.
The data analysis part for swissfel is to be implemented...
"""
import numpy as np


class PsenSetup:
    """Class describing photon spectral encoder (PSEN) setup.
    """

    def __init__(self, ttcalib):
        """Initialize PsenSetup object.

        Args:
            ttcalib: calibration constant of the timing tool in [fs/pixel]
        """
        self.ttcalib = ttcalib
        self.tags = []
        self.data = []

    def load_swissfel_data(self):
        """Load photon spectral encoder (PSEN) data.
        """
        pass

    def load_sacla_data(self, filepath, method='average', remove_nans=True):
        """Load SACLA timing tool data into tags and data.

        Args:
            filepath: path of a csv file with data
            method: {'derivative', 'fitting', 'average'(default)} available methods of timing edge
                detection
            remove_nans: clear data from NaNs also removing the corresponding tags
        """
        data = np.genfromtxt(filepath, delimiter=',', skip_header=2)
        self.tags = data[:, 0]  # first column is a tags list

        if method == 'derivative':
            data = data[:, 1]  # second column are timing edge derivative values (pixel)
        elif method == 'fitting':
            data = data[:, 2]  # first column are timing edge fitting values (pixel)
        elif method == 'average':
            data = (data[:, 1] + data[:, 2]) / 2
        else:
            raise RuntimeError("Method '{}' is not recognised".format(method))

        self.data = data*self.ttcalib  # convert to fs

        if remove_nans:
            idx = ~np.isnan(data)
            self.tags = self.tags[idx]
            self.data = self.data[idx]
