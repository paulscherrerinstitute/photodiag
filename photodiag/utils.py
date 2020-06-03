import json
import warnings

import h5py
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
        x_known=np.arange(data_length),
        x_interp=np.arange(0, data_length - 1, refinement),
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


def savgol_filter_1d(data, period, window, steps):
    C = 2.99792458
    freq = C / np.linspace(*window, steps)
    freq_interp = np.linspace(C / window[1], C / window[0], steps)

    tmp = np.interp(freq_interp, freq[::-1], data[::-1])
    tmp2 = signal.savgol_filter(tmp, period, 1)
    data_out = np.interp(freq, freq_interp, tmp2)

    return data_out


def savgol_filter(data, period, window, steps):
    C = 2.99792458
    freq = C / np.linspace(*window, steps)
    freq_interp = np.linspace(C / window[1], C / window[0], steps)

    tmp = np.apply_along_axis(_interpolate_row, 0, data[::-1], freq[::-1], freq_interp)
    tmp2 = signal.savgol_filter(tmp, period, 1)
    data_out = np.apply_along_axis(_interpolate_row, 0, tmp2, freq_interp, freq)

    return data_out


def _interpolate_row(y_known, x_known, x_interp):
    y_interp = np.interp(x_interp, x_known, y_known)
    return y_interp


def read_bsread_file(filepath, signal_channel, events_channel, dark_shot_event, dark_shot_filter):
    """Read encoder data from bsread hdf5 file.
    """
    with h5py.File(filepath, "r") as h5f:
        if "/data" in h5f:
            # sf_databuffer_writer format
            path_prefix = "/data/{}"
        else:
            # bsread format
            path_prefix = "/{}"

        signal_channel_group = h5f[path_prefix.format(signal_channel)]
        signal_pulse_id = signal_channel_group["pulse_id"][:]

        if events_channel:
            events_channel_group = h5f[path_prefix.format(events_channel)]
            events_pulse_id = events_channel_group["pulse_id"][:]

            pid, index, event_index = np.intersect1d(
                signal_pulse_id, events_pulse_id, return_indices=True
            )

            # if both groups have 0 in their pulse_id
            pid_zero_ind = pid == 0
            if any(pid_zero_ind):
                warnings.warn(
                    f"\n \
                File: {filepath}\n \
                Both '{signal_channel}' and '{events_channel}' have zeroed pulse_id(s).\n"
                )
                index = index[~pid_zero_ind]
                event_index = event_index[~pid_zero_ind]

            is_dark = events_channel_group["data"][event_index, dark_shot_event].astype(bool)

        elif dark_shot_filter:
            index = signal_pulse_id != 0
            is_dark = dark_shot_filter(signal_pulse_id)[index]

        else:
            index = signal_pulse_id != 0
            is_dark = None

        signal_pulse_id = signal_pulse_id[index]

        # data is stored as uint16 in hdf5, so has to be casted to float for further analysis,
        images = signal_channel_group["data"][index].astype(float)

        # averaging every image over y-axis gives the final raw waveforms
        data = images.mean(axis=1)

    return data, signal_pulse_id, is_dark
