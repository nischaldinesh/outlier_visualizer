# detectors.py
from __future__ import annotations
from typing import Tuple, Optional, Literal
import numpy as np

# Optional import of PyOD; we error clearly in caller if not installed.
try:
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    from pyod.models.mcd import MCD
    from pyod.models.ocsvm import OCSVM
    _PYOD_OK = True
except Exception:
    _PYOD_OK = False


def ensure_pyod() -> None:
    if not _PYOD_OK:
        raise ImportError(
            "PyOD is required for CBLOF/HBOS/MCD/OCSVM.\n"
            "Install with: pip install pyod"
        )


def get_detector(
    name: Literal["CBLOF", "HBOS", "MCD", "OCSVM"],
    contamination: float = 0.05,
    random_state: Optional[int] = None,
) -> object:
    """
    Return an unfitted PyOD detector with reasonable defaults.
    """
    ensure_pyod()
    name = name.upper()
    if name == "CBLOF":
        # Small, robust defaults
        return CBLOF(contamination=contamination, random_state=random_state, check_estimator=False)
    if name == "HBOS":
        return HBOS(contamination=contamination)
    if name == "MCD":
        return MCD(contamination=contamination, random_state=random_state)
    if name == "OCSVM":
        # RBF kernel, auto gamma works okay on standardized data
        return OCSVM(contamination=contamination, kernel="rbf", gamma="auto")
    raise ValueError(f"Unknown detector: {name}")


def fit_and_score(detector, X: np.ndarray) -> np.ndarray:
    """
    Fit detector on X and return per-sample anomaly scores (higher => more anomalous).
    PyOD convention: decision_scores_ (train) and decision_function(X) (predict) are higher for outliers.
    """
    detector.fit(X)
    # If we trained on X and want train-scores for the same points:
    if hasattr(detector, "decision_scores_"):
        # Normalize sign so higher = more anomalous
        scores = np.asarray(detector.decision_scores_, dtype=float)
    else:
        scores = detector.decision_function(X)
    return scores


def binned_score_heatmap(
    X2d: np.ndarray,
    scores: np.ndarray,
    bins: int = 60,
    technique: Literal["Raw", "Threshold", "Interpolated", "Binary", "Ranked"] = "Raw",
    threshold_q: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Convert point scores into a 2D grid heatmap over the t-SNE plane using mean score per bin.

    Returns:
      Z: (ny, nx) grid of values
      xedges, yedges: bin edges
      zsmooth_flag: whether to request interpolation in the front-end (Plotly Heatmap's zsmooth).
    """
    x = X2d[:, 0]
    y = X2d[:, 1]
    # Weighted mean per bin
    sum_scores, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=scores)
    counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    with np.errstate(invalid="ignore"):
        Z = sum_scores / np.maximum(counts, 1.0)

    # Normalize to [0,1] (robust)
    finite_mask = np.isfinite(Z)
    if np.any(finite_mask):
        zmin = np.nanpercentile(Z[finite_mask], 1)
        zmax = np.nanpercentile(Z[finite_mask], 99)
        if zmax > zmin:
            Z = (Z - zmin) / (zmax - zmin)
        else:
            Z = np.zeros_like(Z)

    zsmooth = False

    if technique == "Threshold":
        thr = np.nanpercentile(Z[finite_mask], threshold_q) if np.any(finite_mask) else 1.0
        Z = np.where(Z >= thr, Z, np.nan)
    elif technique == "Interpolated":
        # Use same Z; front-end will request smoothing
        zsmooth = True
    elif technique == "Binary":
        thr = np.nanpercentile(Z[finite_mask], threshold_q) if np.any(finite_mask) else 1.0
        Z = np.where(Z >= thr, 1.0, 0.0)
    elif technique == "Ranked":
        # Rank-normalize finite values to [0,1]
        vals = Z[finite_mask]
        order = np.argsort(vals)
        ranks = np.empty_like(vals)
        ranks[order] = np.linspace(0, 1, num=len(vals), endpoint=True)
        Z2 = np.full_like(Z, np.nan)
        Z2[finite_mask] = ranks
        Z = Z2

    return Z.T, xedges, yedges, zsmooth
