import json

import numpy as np
from scipy import signal


def read_eco_scan(filepath):
    """Extract `scan_readbacks` and corresponding bsread `scan_files` from an eco scan.

    Args:
        filepath: path to a json eco scan file to read data from
    Returns:
        scan_pos_fs, bsread_files
    """
    with open(filepath) as eco_file:
        eco_scan = json.load(eco_file)

    # flatten scan_readbacks array and convert values to femtoseconds
    scan_pos_fs = np.ravel(eco_scan["scan_readbacks"]) * 1e15

    scan_files = eco_scan["scan_files"]
    # bsread file is 'normally' a first file on a list, but maybe the following should be
    # implemented in a more robust way
    bsread_files = [scan_file[0] for scan_file in scan_files]

    return scan_pos_fs, bsread_files


def find_edge(data, step_length=50, edge_type="falling", refinement=1):
    # refine data
    data_length = data.shape[1]
    refined_data = np.apply_along_axis(
        _interpolate_row,
        axis=1,
        arr=data,
        interp_energy=np.arange(0, data_length - 1, refinement),
        energy=np.arange(data_length),
    )

    # prepare a step function and refine it
    step_waveform = np.ones(shape=(step_length,))
    if edge_type == "rising":
        step_waveform[: int(step_length / 2)] = -1
    elif edge_type == "falling":
        step_waveform[int(step_length / 2) :] = -1

    step_waveform = np.interp(
        x=np.arange(0, step_length - 1, refinement), xp=np.arange(step_length), fp=step_waveform
    )

    # find edges
    xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode="valid")
    edge_position = np.argmax(xcorr, axis=1).astype(float) * refinement
    xcorr_amplitude = np.amax(xcorr, axis=1)

    # correct edge_position for step_length
    edge_position += np.floor(step_length / 2)

    return {"edge_pos": edge_position, "xcorr": xcorr, "xcorr_ampl": xcorr_amplitude}


def savgol_filter(data, period, window, steps):
    C = 2.99792458
    lambda_nm = np.linspace(*window, steps)
    freq = C / lambda_nm
    interp_freq = np.linspace(C / window[1], C / window[0], steps)

    tmp = np.apply_along_axis(_interpolate_row, 0, data[::-1], freq[::-1], interp_freq)[::-1]
    tmp2 = signal.savgol_filter(tmp, period, 1)
    data_out = np.apply_along_axis(_interpolate_row, 0, tmp2[::-1], interp_freq, freq[::-1])[::-1]

    return data_out


def _interpolate_row(data, energy, interp_energy):
    return np.interp(interp_energy, energy, data)
