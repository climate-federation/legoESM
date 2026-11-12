"""Measure WW3 agreement on the actual common times, cells and station spectra."""

import numpy as np


def errors(prediction, reference, valid):
    """Float64 reductions; an identically zero reference has no relative metric."""
    prediction, reference = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    valid = np.broadcast_to(valid, prediction.shape)
    if not np.any(valid):
        return dict(count=0, relative_l2=None, peak_normalized=None, rmse=None, max_absolute=None)
    if not np.isfinite(prediction[valid]).all() or not np.isfinite(reference[valid]).all():
        raise ValueError("Nonfinite values in the declared comparison domain")
    delta = np.where(valid, prediction - reference, 0)
    norm = np.linalg.norm(reference[valid].ravel())
    peak = np.max(abs(reference[valid]))
    location = np.unravel_index(np.argmax(np.where(valid, abs(delta), -np.inf)), delta.shape)
    absolute = float(abs(delta[location]))
    return dict(count=int(valid.sum()), relative_l2=float(np.linalg.norm(delta.ravel()) / norm) if norm else None,
                peak_normalized=absolute / float(peak) if peak else None,
                rmse=float(np.sqrt(np.mean(delta[valid]**2))), max_absolute=absolute,
                max_absolute_index=[int(i) for i in location])


def circular_errors(prediction, reference, valid):
    difference = (prediction - reference + 180) % 360 - 180
    return errors(difference, np.zeros_like(difference), valid)
