"""Evaluation helpers: RMSE computation and sequence alignment helpers.

Small, dependency-free utilities to compute masked log-RMSE and align
sequence-model predictions to the full timeline for fair comparison.
"""
from typing import Optional, Tuple
import numpy as np


def compute_log_rmse(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    """Compute RMSE in log-space for arrays with no NaNs expected.

    Both inputs should be numeric numpy arrays of the same shape.
    """
    # Fast path: assume no NaNs
    return float(np.sqrt(np.mean((y_log_true.flatten() - y_log_pred.flatten()) ** 2)))


def compute_log_rmse_masked(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    """Compute RMSE on log-space arrays while ignoring pairs with NaNs.

    Returns np.nan if there are no overlapping non-NaN entries.
    """
    mask = (~np.isnan(y_log_true)) & (~np.isnan(y_log_pred))
    if not np.any(mask):
        return float('nan')
    return float(np.sqrt(np.mean((y_log_true[mask] - y_log_pred[mask]) ** 2)))


def align_sequence_predictions(
    preds: np.ndarray,
    full_shape: Tuple[int, int],
    seq_start: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> np.ndarray:
    """Align sequence-model predictions into a full timeline array.

    preds: [T_eval, N] predictions from a sequence model
    full_shape: (T_total, N) desired output shape
    seq_start: optional start index in the full timeline where preds begin.
               If None, the function will assume preds align after `seq_len` padding
               at the start (i.e., seq_start = seq_len).
    seq_len: optional sequence length used to produce preds. Required if seq_start is None.

    Returns an array shaped `full_shape` with NaNs where no prediction is available.
    """
    T_total, N = full_shape
    Z = np.full(full_shape, np.nan, dtype=float)
    T_eval = preds.shape[0]
    if seq_start is None:
        if seq_len is None:
            raise ValueError("Either seq_start or seq_len must be provided to align sequence predictions")
        seq_start = seq_len

    if seq_start < 0 or seq_start + T_eval > T_total:
        # Clip to available range
        insert_start = max(0, seq_start)
        insert_end = min(T_total, seq_start + T_eval)
        src_start = max(0, -seq_start)
        src_end = src_start + (insert_end - insert_start)
        Z[insert_start:insert_end, :] = preds[src_start:src_end, :]
    else:
        Z[seq_start:seq_start + T_eval, :] = preds
    return Z
