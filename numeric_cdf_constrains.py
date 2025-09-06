# numeric_cdf_constraints.py
# Module to build/validate continuous CDFs compatible with Metaculus.

from __future__ import annotations
import numpy as np


def _project_bounded_simplex(
    d_raw: np.ndarray,
    total: float,
    L: float,
    U: float,
    tol: float = 1e-12,
    max_iter: int = 60,
) -> np.ndarray:
    """
    Euclidean projection of d_raw onto the set:
        { d : sum(d) = total,  L <= d_i <= U }.

    Solved via bisection on the Lagrange multiplier because
    S(λ) = sum(clip(d_raw + λ, L, U)) is monotonic in λ.
    Returns the projected vector that stays as close as possible (L2) to d_raw.

    Technical note: projection onto the “capped simplex”
    (simplex with lower/upper bounds per coordinate).
    """
    d_raw = np.asarray(d_raw, dtype=float)
    n = d_raw.size
    if n == 0:
        return d_raw.copy()

    # Ensure target sum is feasible.
    total = float(np.clip(total, n * L, n * U))

    # Bound λ so that a solution exists within [lo, hi].
    lo = float(np.min(L - d_raw) - 1.0)
    hi = float(np.max(U - d_raw) + 1.0)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = np.clip(d_raw + mid, L, U).sum()
        if abs(s - total) <= tol:
            return np.clip(d_raw + mid, L, U)
        if s > total:
            hi = mid
        else:
            lo = mid

    # Fallback if not converged within max_iter (very rare with these sizes).
    return np.clip(d_raw + 0.5 * (lo + hi), L, U)


def _anti_flatten_postpass(
    cdf_in: np.ndarray,
    open_lower: bool,
    open_upper: bool,
    min_step: float = 5e-05,
    max_step: float = 0.59 - 1e-6,
    cv_thresh: float = 0.10,
    blend: float = 0.20,
) -> np.ndarray:
    """
    If the CDF’s increments are too uniform (low coefficient of variation),
    blend in a Gaussian (bell-shaped) kernel and re-project to respect:
      - total mass (open/closed bounds)
      - per-step limits in [min_step, max_step]

    cv_thresh: flatness threshold; if CV(diff(cdf)) < cv_thresh, apply correction.
    blend:     kernel mixing weight (0..1). 0.20–0.30 works well in practice.

    Returns a CDF with the same length; step constraints remain valid.
    """
    cdf = np.asarray(cdf_in, dtype=float)
    d = np.diff(cdf)
    if d.size == 0:
        return cdf

    mean = d.mean()
    if mean <= 0:
        return cdf

    cv = (d.std() / (abs(mean) + 1e-12))
    if cv >= cv_thresh:
        return cdf  # already enough variation: keep as is

    m = len(d)
    idx = np.arange(m)
    center = 0.5 * (m - 1)
    sigma = max(m / 4.0, 1.0)  # smooth width
    w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    w /= w.sum()

    # Blend while preserving total mass (sum(d)).
    d_tilt = (1.0 - blend) * d + blend * (w * d.sum())

    # Re-project to bounds and correct total according to limits.
    lower = 0.001 if open_lower else 0.0
    upper = 0.999 if open_upper else 1.0
    total = upper - lower

    L = min_step
    if m * L > total:  # ensure feasibility
        L = max(total / m - 1e-12, 0.0)

    d_proj = _project_bounded_simplex(d_tilt, total=total, L=L, U=max_step)

    cdf_out = np.empty_like(cdf)
    cdf_out[0] = lower
    cdf_out[1:] = lower + np.cumsum(d_proj)
    cdf_out[-1] = min(cdf_out[-1], upper)
    return cdf_out


def enforce_cdf_constraints(
    cdf_raw: np.ndarray,
    open_lower: bool,
    open_upper: bool,
    min_step: float = 5e-05,
    max_step: float = 0.59 - 1e-6,
) -> np.ndarray:
    """
    Normalize a CDF (typically 201 points in Metaculus) to satisfy:

      • Non-decreasing monotonicity.
      • PER-STEP increment within [min_step, max_step]
        (e.g., [5e-05, 0.59]); the limit is per step, not divided by the number of points.
      • Open/closed bounds:
          - open lower  → cdf[0] ≥ 0.001
          - open upper  → cdf[-1] ≤ 0.999
          - closed      → 0.0 and 1.0 respectively.
      • Preserve total mass and keep the original increment shape as much as possible.

    It also applies an anti-flatten post-process when increments end up too uniform,
    to avoid a near-uniform pdf (“rectangle” look).

    Notes:
      - Metaculus typically expects continuous CDFs discretized at 201 points.
      - Run this after your initial interpolation (linear, PCHIP, etc.).
    """
    cdf_raw = np.asarray(cdf_raw, dtype=float).copy()
    n = len(cdf_raw)
    if n < 2:
        return cdf_raw

    # 1) Limits according to 'open'
    lower_limit = 0.001 if open_lower else 0.0
    upper_limit = 0.999 if open_upper else 1.0

    # 2) Basic cleanup: clamp to [0,1] and enforce monotonicity
    c = np.clip(cdf_raw, 0.0, 1.0)
    c = np.maximum.accumulate(c)

    # 3) Raw increments and desired mass
    d_raw = np.diff(c)
    d_raw = np.maximum(d_raw, 0.0)  # no negative steps
    total = upper_limit - lower_limit
    m = n - 1

    # 4) Adjust feasible min_step if the range is small
    L = min_step
    if m * L > total:
        L = max(total / m - 1e-12, 0.0)

    # 5) Project onto the bounded simplex: correct sum and per-step bounds
    d_proj = _project_bounded_simplex(d_raw, total=total, L=L, U=max_step)

    # 6) Reconstruct + clamp endpoints
    cdf_fix = np.empty_like(cdf_raw)
    cdf_fix[0] = lower_limit
    cdf_fix[1:] = lower_limit + np.cumsum(d_proj)
    cdf_fix[-1] = min(cdf_fix[-1], upper_limit)

    # 7) Anti-flatten post-pass if increments are still too uniform
    cdf_fix = _anti_flatten_postpass(
        cdf_fix,
        open_lower=open_lower,
        open_upper=open_upper,
        min_step=L,              # use the feasible L we computed
        max_step=max_step,
        cv_thresh=0.10,          # lower to 0.08 if flatness persists
        blend=0.20               # increase to 0.25–0.30 for stronger bell shape
    )

    return cdf_fix


# === Console previews for CDF/pdf (ASCII/Unicode) ===

_BLOCKS = np.array(list("▁▂▃▄▅▆▇█"))

def sparkline(vals) -> str:
    """Inline sparkline using Unicode Block Elements. Good to visualize pdf."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return ""
    # NumPy 2.0 removed ndarray.ptp; use np.ptp(v) instead.
    rng = float(np.ptp(v))  # range = max - min
    if rng <= 0.0 or not np.isfinite(rng):
        # all equal or invalid -> draw a flat line based on the value
        return _BLOCKS[0] * max(1, v.size)
    v = (v - float(v.min())) / (rng + 1e-12)
    idx = np.clip(np.rint(v * (len(_BLOCKS) - 1)).astype(int), 0, len(_BLOCKS) - 1)
    return "".join(_BLOCKS[idx])

def pdf_sparkline_from_cdf(cdf) -> str:
    """Sparkline of the pdf = diff(CDF). A bell-ish shape should appear if OK."""
    c = np.asarray(cdf, dtype=float)
    d = np.diff(np.clip(c, 0.0, 1.0))
    return sparkline(d)

def ascii_plot_cdf(cdf, width: int = 80, height: int = 16, y_ticks=(0.0, 0.5, 1.0)) -> None:
    """
    2D ASCII plot of the CDF; draws markers at the discretized CDF heights.
    width/height only affect console rendering (not your real CDF).
    """
    c = np.asarray(cdf, dtype=float)
    # Resample to columns
    xs = np.linspace(0, len(c) - 1, width)
    ys = np.interp(xs, np.arange(len(c)), c)
    # Canvas
    H, W = height, width
    grid = [[" "] * W for _ in range(H)]
    # Draw CDF points
    for j, val in enumerate(ys):
        r = int(round((1.0 - val) * (H - 1)))  # 0=top
        r = max(0, min(H - 1, r))
        grid[r][j] = "█"
    # Grid lines for y_ticks
    for t in y_ticks:
        r = int(round((1.0 - t) * (H - 1)))
        if 0 <= r < H:
            for j in range(W):
                if grid[r][j] == " ":
                    grid[r][j] = "─"
    # Print with labels
    lines = []
    for i, row in enumerate(grid):
        yval = 1.0 - i / (H - 1)
        lines.append(f"{yval:4.2f} │ " + "".join(row))
    lines.append("     └" + "─" * (W - 1))
    print("\n".join(lines))

def cdf_diagnostics(cdf) -> None:
    """Quick stats to catch validation issues and flat shapes."""
    d = np.diff(np.asarray(cdf, dtype=float))
    if d.size == 0:
        print("CDF diag — empty CDF.")
        return
    mean = d.mean()
    cv = (d.std() / (abs(mean) + 1e-12)) if np.isfinite(mean) else float("nan")
    print(
        "CDF diag — steps:",
        f"min={d.min():.6f}, max={d.max():.6f}, mean={mean:.6f}, CV={cv:.3f},",
        f"sum(d)={d.sum():.6f}, ends=({float(cdf[0]):.3f},{float(cdf[-1]):.3f})"
    )