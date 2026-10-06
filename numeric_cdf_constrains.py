# numeric_cdf_constraints.py
# Module to build/validate continuous CDFs compatible with Metaculus.

from __future__ import annotations

import numpy as np

# Metaculus numeric questions discretize the CDF at 201 points; discrete
# questions at inbound_outcome_count + 1 points. The API scales its per-step
# limits by the inbound outcome count (questions/serializers/common.py in
# Metaculus/metaculus), so they must be derived from the CDF length, not
# hardcoded: a 4-outcome discrete question allows steps up to 10, and the
# numeric 0.2 cap would make a closed upper bound unreachable.
DEFAULT_INBOUND_OUTCOME_COUNT = 200


def server_step_bounds(cdf_size: int) -> tuple[float, float]:
    """
    Per-step (min, max) increment the Metaculus API accepts for a CDF of
    cdf_size points. Numeric questions (201 points) get 5e-05..0.2; a
    4-outcome discrete question gets 0.0025..10.
    """
    inbound = max(cdf_size - 1, 1)
    return 0.01 / inbound, 0.2 * DEFAULT_INBOUND_OUTCOME_COUNT / inbound


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
    lower: float,
    upper: float,
    min_step: float,
    max_step: float,
    cv_thresh: float = 0.10,
    blend: float = 0.20,
) -> np.ndarray:
    """
    If the CDF’s increments are too uniform (low coefficient of variation),
    blend in a Gaussian (bell-shaped) kernel and re-project to respect:
      - total mass (the given endpoint masses, e.g. elicited open-bound tails)
      - per-step limits in [min_step, max_step]

    cv_thresh: flatness threshold; if CV(diff(cdf)) < cv_thresh, apply correction.
    blend:     kernel mixing weight (0..1). 0.20–0.30 works well in practice.

    min_step/max_step must be the server_step_bounds for this CDF length.

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
    min_step: float | None = None,
    max_step: float | None = None,
) -> np.ndarray:
    """
    Normalize a CDF to satisfy everything the Metaculus API validates:

      • Non-decreasing monotonicity.
      • PER-STEP increment within the server's scaled limits
        (5e-05..0.2 for the 201-point numeric CDF; wider for short
        discrete CDFs — see server_step_bounds).
      • Open/closed bounds:
          - open lower  → cdf[0] ≥ 0.001
          - open upper  → cdf[-1] ≤ 0.999
          - closed      → exactly 0.0 and 1.0 respectively (the API compares
            with ==, so endpoints are snapped bit-exact at the end).
      • PRESERVE the endpoint masses of the input: 0.001/0.999 are API validity
        floors, not targets. A CDF that already leaves real mass outside an
        open bound (e.g. 5%) keeps it; only degenerate endpoints are moved.

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

    if min_step is None or max_step is None:
        derived_min, derived_max = server_step_bounds(n)
        min_step = derived_min if min_step is None else min_step
        max_step = derived_max if max_step is None else max_step

    # 1) Basic cleanup: clamp to [0,1] and enforce monotonicity
    c = np.clip(cdf_raw, 0.0, 1.0)
    c = np.maximum.accumulate(c)

    # 2) Target endpoints. On open bounds the 0.001/0.999 API limits act as
    #    floors/caps only: healthy elicited tail masses pass through untouched.
    if open_lower:
        lower_target = float(np.clip(c[0], 0.001, 0.999))
    else:
        lower_target = 0.0
    if open_upper:
        upper_target = float(np.clip(c[-1], 0.001, 0.999))
    else:
        upper_target = 1.0
    if upper_target - lower_target < 1e-6:
        # Degenerate input (no usable span): fall back to the maximal API span
        lower_target = 0.001 if open_lower else 0.0
        upper_target = 0.999 if open_upper else 1.0

    # 3) Raw increments and desired mass
    d_raw = np.diff(c)
    d_raw = np.maximum(d_raw, 0.0)  # no negative steps
    total = upper_target - lower_target
    m = n - 1

    # 4) Adjust feasible min_step if the range is small
    L = min_step
    if m * L > total:
        L = max(total / m - 1e-12, 0.0)

    # 5) Project onto the bounded simplex: correct sum and per-step bounds
    d_proj = _project_bounded_simplex(d_raw, total=total, L=L, U=max_step)

    # 6) Reconstruct + clamp endpoints
    cdf_fix = np.empty_like(cdf_raw)
    cdf_fix[0] = lower_target
    cdf_fix[1:] = lower_target + np.cumsum(d_proj)
    cdf_fix[-1] = min(cdf_fix[-1], upper_target)

    # 7) Anti-flatten post-pass if increments are still too uniform
    cdf_fix = _anti_flatten_postpass(
        cdf_fix,
        lower=lower_target,
        upper=upper_target,
        min_step=L,              # the feasible min we computed
        max_step=max_step,
        cv_thresh=0.10,          # lower to 0.08 if flatness persists
        blend=0.20               # increase to 0.25–0.30 for stronger bell shape
    )

    # 8) Bit-exact endpoints on closed bounds: the API compares cdf[0]/cdf[-1]
    # with == (0.0/1.00). The projections above land within ~1e-12 of the
    # target; snap so the endpoint survives the JSON round-trip exactly.
    if not open_lower:
        cdf_fix[0] = 0.0
    if not open_upper:
        cdf_fix[-1] = 1.0

    return cdf_fix


def validate_cdf_for_submission(
    cdf: list[float] | np.ndarray,
    open_lower: bool,
    open_upper: bool,
    inbound_outcome_count: int | None = None,
) -> None:
    """
    Client-side mirror of the Metaculus API's CDF validation
    (continuous_validation in questions/serializers/common.py), so an
    invalid CDF fails here with a clear message instead of as a 400 after
    the research and forecast runs have already been spent. Raises
    ValueError with the same complaints the server would raise; silent
    return means the server will accept it.
    """
    if inbound_outcome_count is None or inbound_outcome_count <= 0:
        inbound_outcome_count = DEFAULT_INBOUND_OUTCOME_COUNT

    cdf = np.round(np.asarray(cdf, dtype=float), 10).tolist()
    if len(cdf) < 2:
        raise ValueError(
            f"CDF Invalid:\ncontinuous_cdf must have "
            f"{inbound_outcome_count + 1} values (got {len(cdf)}).\n"
        )
    steps = np.round(np.diff(cdf), 9)

    errors = ""
    if len(cdf) != inbound_outcome_count + 1:
        errors += (
            f"continuous_cdf must have {inbound_outcome_count + 1} values "
            f"(got {len(cdf)}).\n"
        )
    min_diff = np.round(0.01 / inbound_outcome_count, 9)
    if not np.all(steps >= min_diff):
        errors += (
            "continuous_cdf must be increasing by at least "
            f"{min_diff} at every step.\n"
        )
    max_diff = 0.2 * DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
    if not np.all(steps <= max_diff):
        errors += (
            "continuous_cdf must be increasing by no more than "
            f"{max_diff} at every step.\n"
        )
    if open_lower:
        if not cdf[0] >= 0.001:
            errors += (
                "continuous_cdf at lower bound must be at least 0.001 "
                "due to lower bound being open.\n"
            )
    elif cdf[0] != 0.00:
        errors += "continuous_cdf[0] must be 0.0 (closed lower bound).\n"
    if open_upper:
        if not cdf[-1] <= 0.999:
            errors += (
                "continuous_cdf at upper bound must be at most 0.999 "
                "due to upper bound being open.\n"
            )
    elif cdf[-1] != 1.00:
        errors += (
            "continuous_cdf at upper bound must be 1.00 "
            "due to upper bound being closed.\n"
        )
    if errors:
        raise ValueError("CDF Invalid:\n" + errors)


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
        r = round((1.0 - val) * (H - 1))  # 0=top
        r = max(0, min(H - 1, r))
        grid[r][j] = "█"
    # Grid lines for y_ticks
    for t in y_ticks:
        r = round((1.0 - t) * (H - 1))
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