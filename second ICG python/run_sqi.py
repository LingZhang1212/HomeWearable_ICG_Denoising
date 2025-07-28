import numpy as np
from scipy.spatial import cKDTree


def run_sqi(refqrs, testqrs, thres=0.05, margin=2, windowlen=60, fs=1000):
    """
    Compare two sets of annotations: a reference and a test.
    
    Parameters:
    - refqrs:     reference QRS annotations (in seconds)
    - testqrs:    test QRS annotations (in seconds)
    - thres:      threshold in seconds for matching beats
    - margin:     time margin to exclude from start and end (in seconds)
    - windowlen:  length of comparison window (in seconds)
    - fs:         sampling frequency (Hz)

    Returns:
    - F1: F1-score
    - Se: Sensitivity
    - PPV: Positive Predictive Value
    - Nb: Dictionary with TP, FN, FP counts
    """
    try:
        refqrs = np.asarray(refqrs).flatten()
        testqrs = np.asarray(testqrs).flatten()

        # Convert from seconds to samples
        refqrs = refqrs * fs
        testqrs = testqrs * fs

        # Remove annotations outside the evaluation window
        start = margin * fs
        stop = (windowlen - margin) * fs
        refqrs = refqrs[(refqrs > start) & (refqrs < stop)]
        testqrs = testqrs[(testqrs > start) & (testqrs < stop)]

        if len(refqrs) == 0:
            return None, None, None, {}

        # Handle borders for refqrs
        border_inds = np.where((refqrs < thres * fs) | (refqrs > (windowlen - thres) * fs))[0]
        if len(border_inds) > 0:
            tree = cKDTree(testqrs.reshape(-1, 1))
            dists, _ = tree.query(refqrs[border_inds].reshape(-1, 1), k=1)
            keep = dists < thres * fs
            refqrs = np.delete(refqrs, border_inds[~keep])

        # Handle borders for testqrs
        border_inds = np.where((testqrs < thres * fs) | (testqrs > (windowlen - thres) * fs))[0]
        if len(border_inds) > 0:
            tree = cKDTree(refqrs.reshape(-1, 1))
            dists, _ = tree.query(testqrs[border_inds].reshape(-1, 1), k=1)
            keep = dists < thres * fs
            testqrs = np.delete(testqrs, border_inds[~keep])

        # Core comparison
        ref_tree = cKDTree(refqrs.reshape(-1, 1))
        dists, indices = ref_tree.query(testqrs.reshape(-1, 1), k=1)
        matched_ref_indices = indices[dists < thres * fs]
        TP = len(np.unique(matched_ref_indices))
        FN = len(refqrs) - TP
        FP = len(testqrs) - TP

        Se = TP / (TP + FN) if (TP + FN) > 0 else 0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1 = 2 * Se * PPV / (Se + PPV) if (Se + PPV) > 0 else 0

        Nb = {'TP': TP, 'FN': FN, 'FP': FP}
        return F1, Se, PPV, Nb

    except Exception:
        return None, None, None, {}

