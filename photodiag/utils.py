import numpy as np


def find_edge(data, step_length, edge_type, refinement):
    # refine data
    def _interp(fp, xp, x):  # utility function to be used with apply_along_axis
        return np.interp(x, xp, fp)

    data_length = data.shape[1]
    refined_data = np.apply_along_axis(
        _interp,
        axis=1,
        arr=data,
        x=np.arange(0, data_length - 1, refinement),
        xp=np.arange(data_length),
    )

    # prepare a step function and refine it
    step_waveform = np.ones(shape=(step_length,))
    if edge_type == 'rising':
        step_waveform[: int(step_length / 2)] = -1
    elif edge_type == 'falling':
        step_waveform[int(step_length / 2) :] = -1

    step_waveform = np.interp(
        x=np.arange(0, step_length - 1, refinement),
        xp=np.arange(step_length),
        fp=step_waveform,
    )

    # find edges
    xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode='valid')
    edge_position = np.argmax(xcorr, axis=1).astype(float) * refinement
    xcorr_amplitude = np.amax(xcorr, axis=1)

    # correct edge_position for step_length
    edge_position += np.floor(step_length / 2)

    return {'edge_pos': edge_position, 'xcorr': xcorr, 'xcorr_ampl': xcorr_amplitude}
