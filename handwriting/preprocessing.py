"""
handwriting/preprocessing.py - one path from raw pen samples to Transformer input,
shared by training (EMOTHAW, via the importer's standard CSVs) and inference
(browser canvas).

    raw points  (x, y, t, pen, pressure?, altitude?, azimuth?)
        -> Trajectory in canvas convention (x right, y DOWN, t seconds)
        -> clean: drop non-finite rows, stable sort by time
        -> trim in-air points before the first / after the last pen-down point
        -> resample to <= RESAMPLE_HZ (pen-state changes always kept, so stroke
           boundaries survive)
        -> per-point features (config.KINEMATIC_FEATURES + pen features)
        -> standardise with the TRAINING-set mean/std stored in the checkpoint
        -> windows of <= MAX_LEN tokens

Stroke boundaries come from the pen state: every pen-down run is one stroke,
in-air points between strokes are kept (with their timing), and the temporal
order is never changed.
"""

import math

import numpy as np

from . import config

_OPTIONAL = ("pressure", "altitude", "azimuth")


class HandwritingInputError(ValueError):
    """The input cannot be analysed; the message is meant for the user."""


class Trajectory:
    """Raw samples in canvas convention. Optional channels are NaN when unavailable."""

    def __init__(self, x, y, t, pen, pressure=None, altitude=None, azimuth=None):
        n = len(x)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)          # seconds
        self.pen = np.asarray(pen, dtype=np.float64)      # 1 down, 0 up
        nan = np.full(n, np.nan)
        self.pressure = nan.copy() if pressure is None else np.asarray(pressure, dtype=np.float64)  # 0..1
        self.altitude = nan.copy() if altitude is None else np.asarray(altitude, dtype=np.float64)  # degrees 0..90
        self.azimuth = nan.copy() if azimuth is None else np.asarray(azimuth, dtype=np.float64)    # degrees 0..360
        for name in ("y", "t", "pen", "pressure", "altitude", "azimuth"):
            if len(getattr(self, name)) != n:
                raise HandwritingInputError(f"'{name}' has a different length from 'x'")

    def __len__(self):
        return len(self.x)

    def take(self, idx):
        return Trajectory(self.x[idx], self.y[idx], self.t[idx], self.pen[idx],
                          self.pressure[idx], self.altitude[idx], self.azimuth[idx])


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
# Raw .svc columns (Wacom, as distributed with EMOTHAW).
SVC_COLUMNS = ["x", "y", "timestamp", "pen_status", "azimuth", "altitude", "pressure"]
# Full-scale raw values used to normalise the pen channels. The importer checks
# every file against them and stops if the data exceeds them.
SVC_PRESSURE_MAX = 1023.0        # 10-bit pressure
SVC_AZIMUTH_MAX = 4095.0         # -> 0..360 deg (normalisation given in the EMOTHAW paper)
SVC_ALTITUDE_MAX = 1023.0        # -> 0..90 deg


def read_svc_rows(path):
    """.svc: first line = number of points, then rows of
    x y timestamp(ms) pen_status azimuth altitude pressure (raw tablet units).
    -> (N, 7) float array in file order."""
    with open(path) as f:
        lines = f.read().split("\n")
    try:
        n = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"{path}: first line must be the number of points, got {lines[0][:40]!r}") from None
    rows = np.array([[float(v) for v in line.split()] for line in lines[1:n + 1] if line.strip()],
                    dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"{path}: expected 7 columns, got shape {rows.shape}")
    return rows


def svc_to_trajectory(rows, pressure_max=SVC_PRESSURE_MAX, azimuth_max=SVC_AZIMUTH_MAX,
                      altitude_max=SVC_ALTITUDE_MAX):
    """Raw .svc rows -> Trajectory in canvas convention.

    The EMOTHAW tablet was used rotated by 90 degrees: plotting (x, y) directly
    gives mirrored, sideways writing. Upright writing in canvas convention is
    canvas_x = -tablet_y, canvas_y = -tablet_x (checked by rendering the
    capital-letter and cursive tasks of several writers).
    """
    return Trajectory(
        x=-rows[:, 1], y=-rows[:, 0], t=rows[:, 2] / 1000.0, pen=(rows[:, 3] > 0).astype(float),
        pressure=rows[:, 6] / pressure_max,
        azimuth=rows[:, 4] * 360.0 / azimuth_max,
        altitude=rows[:, 5] * 90.0 / altitude_max,
    )


def read_svc(path):
    return svc_to_trajectory(read_svc_rows(path))


# Standard per-sample CSV written by handwriting.import_emothaw. Canvas
# convention (y down); pressure 0..1, altitude 0..90 deg, azimuth 0..360 deg,
# empty where not recorded.
STANDARD_COLUMNS = ["x", "y", "t_ms", "pen_down", "pressure", "altitude", "azimuth", "stroke"]


def read_trajectory_csv(path):
    """Standard per-sample CSV -> Trajectory. pressure / altitude / azimuth are optional columns."""
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in ("x", "y", "t_ms", "pen_down")):
        raise ValueError(f"{path}: expected columns {STANDARD_COLUMNS}")
    col = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    opt = lambda k: col(k) if k in rows[0] else None
    return Trajectory(x=col("x"), y=col("y"), t=col("t_ms") / 1000.0, pen=col("pen_down"),
                      pressure=opt("pressure"), altitude=opt("altitude"), azimuth=opt("azimuth"))


def _number(value, field, index, required=True):
    if value is None:
        if required:
            raise HandwritingInputError(f"point {index}: '{field}' is missing")
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandwritingInputError(f"point {index}: '{field}' must be a number")
    return float(value)


def from_browser_points(points):
    """[{x, y, t (ms), pen_down, pressure?, altitude? (deg), azimuth? (deg)}] -> Trajectory.

    x/y are canvas-relative CSS pixels (y down). Optional channels may be null.
    `timestamp` is accepted as an alias of `t`.
    """
    if not isinstance(points, list):
        raise HandwritingInputError("'points' must be a list")
    if not points:
        raise HandwritingInputError("Please write something before analyzing.")
    if len(points) > config.MAX_INPUT_POINTS:
        raise HandwritingInputError(f"too many points ({len(points)} > {config.MAX_INPUT_POINTS})")
    cols = {k: np.empty(len(points)) for k in ("x", "y", "t", "pen") + _OPTIONAL}
    for i, p in enumerate(points):
        if not isinstance(p, dict):
            raise HandwritingInputError(f"point {i} must be an object")
        cols["x"][i] = _number(p.get("x"), "x", i)
        cols["y"][i] = _number(p.get("y"), "y", i)
        cols["t"][i] = _number(p.get("t", p.get("timestamp")), "t", i) / 1000.0
        pen = p.get("pen_down")
        if not isinstance(pen, bool):
            raise HandwritingInputError(f"point {i}: 'pen_down' must be true or false")
        cols["pen"][i] = 1.0 if pen else 0.0
        for k in _OPTIONAL:
            cols[k][i] = _number(p.get(k), k, i, required=False)
    return Trajectory(**cols)


# ---------------------------------------------------------------------------
# Cleaning, trimming, resampling
# ---------------------------------------------------------------------------
def clean(traj):
    """Drop rows with non-finite x/y/t, sort by time (stable), clamp optional channels."""
    ok = np.isfinite(traj.x) & np.isfinite(traj.y) & np.isfinite(traj.t)
    traj = traj.take(np.flatnonzero(ok))
    traj = traj.take(np.argsort(traj.t, kind="stable"))
    # Out-of-range optional values are treated as "not reported".
    traj.pressure = np.where((traj.pressure >= 0) & (traj.pressure <= 1), traj.pressure, np.nan)
    traj.altitude = np.where((traj.altitude >= 0) & (traj.altitude <= 90), traj.altitude, np.nan)
    traj.azimuth = np.where((traj.azimuth >= 0) & (traj.azimuth <= 360), traj.azimuth, np.nan)
    return traj


def trim(traj):
    """Keep from the first to the last pen-down point (hover before/after writing is noise)."""
    down = np.flatnonzero(traj.pen > 0)
    if down.size == 0:
        return traj.take(np.zeros(0, dtype=int))
    return traj.take(np.arange(down[0], down[-1] + 1))


def resample(traj, hz=config.RESAMPLE_HZ):
    """Keep at most one point per 1/hz s, plus the first and last point of every
    pen-state run, so stroke starts/ends are never lost."""
    n = len(traj)
    if n <= 2:
        return traj
    step = 1.0 / hz
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    change = np.flatnonzero(np.diff(traj.pen) != 0)
    keep[change] = True          # last point of a run
    keep[change + 1] = True      # first point of the next run
    last_t = traj.t[0]
    for i in range(1, n):
        if keep[i] or traj.t[i] - last_t >= step - 1e-9:
            keep[i] = True
            last_t = traj.t[i]
    return traj.take(np.flatnonzero(keep))


def collapse_air(traj):
    """Keep only the first point of every in-air run (the pen-lift point), for
    models trained on pens that record nothing while hovering."""
    pen = traj.pen > 0
    keep = pen | np.concatenate([[True], pen[:-1]])
    return traj.take(np.flatnonzero(keep))


def _pen_runs(pen):
    """[(start, end)) index ranges of consecutive pen-down points."""
    down = np.concatenate([[False], pen > 0, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(down))
    return list(zip(edges[0::2], edges[1::2]))


def resample_arclength(traj, step=config.ARCLENGTH_STEP):
    """Resample every pen-down stroke at a fixed arc-length spacing of
    step x (median stroke height); in-air points are kept unchanged.

    Used by models whose data has no reliable timing (handwriting reconstructed
    from images): point density then depends only on the shape of the writing,
    not on pixel size or writing speed. t is interpolated along each stroke, so
    real timestamps stay real and an image's sequence index stays an index.
    """
    runs = _pen_runs(traj.pen)
    if not runs:
        return traj
    heights = [np.ptp(traj.y[a:b]) for a, b in runs if b - a > 1]
    down = traj.pen > 0
    extent = max(np.ptp(traj.x[down]), np.ptp(traj.y[down]), 1e-9)
    scale = max(float(np.median(heights)) if heights else 0.0, extent / 300.0)
    spacing = step * scale
    chans = ("x", "y", "t", "pen", "pressure", "altitude", "azimuth")
    out = {k: [] for k in chans}
    prev = 0
    for a, b in runs + [(len(traj), len(traj))]:
        for k in chans:                                    # in-air points before this stroke
            out[k].append(getattr(traj, k)[prev:a])
        if a == b:
            break
        xs, ys = traj.x[a:b], traj.y[a:b]
        seg = np.hypot(np.diff(xs), np.diff(ys))
        keep = np.concatenate([[True], seg > 0])           # drop repeated positions
        idx = np.arange(a, b)[keep]
        s = np.concatenate([[0.0], np.cumsum(seg[seg > 0])])
        targets = np.arange(0.0, s[-1], spacing) if s[-1] > 0 else np.zeros(1)
        if s[-1] > 0 and s[-1] - targets[-1] > 1e-9:
            targets = np.append(targets, s[-1])            # always keep the stroke end
        for k in chans:
            v = getattr(traj, k)[idx]
            out[k].append(np.interp(targets, s, v) if len(s) > 1 else v[:1].copy())
        prev = b
    cols = {k: np.concatenate(v) for k, v in out.items()}
    cols["pen"] = (cols["pen"] > 0.5).astype(float)
    return Trajectory(**cols)


def prepare(traj, air="keep", resample_mode="time"):
    """clean -> [collapse air] -> trim -> resample, with user-facing validation.
    air: "keep" or "collapse"; resample_mode: "time" (<= RESAMPLE_HZ) or
    "arclength" (see resample_arclength). Both come from the checkpoint at inference."""
    traj = clean(traj)
    if air == "collapse":
        traj = collapse_air(traj)
    if int((traj.pen > 0).sum()) == 0:
        raise HandwritingInputError("Please write something before analyzing.")
    if int((traj.pen > 0).sum()) < config.MIN_PEN_DOWN_POINTS:
        raise HandwritingInputError("That is too little handwriting to analyse - please write at "
                                    "least a word or two.")
    traj = resample_arclength(trim(traj)) if resample_mode == "arclength" else resample(trim(traj))
    if len(traj) < config.MIN_TOKENS or traj.t[-1] - traj.t[0] <= 0:
        raise HandwritingInputError("That is too little handwriting to analyse - please write at "
                                    "least a word or two.")
    return traj


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def stroke_ids(pen):
    """Stroke index per point: increments at every pen-down run start; in-air points
    get the index of the stroke they follow (-1 before the first)."""
    starts = (pen > 0) & np.concatenate([[True], pen[:-1] <= 0])
    return np.cumsum(starts) - 1, starts


def compute_features(traj):
    """Trajectory (already prepared) -> (features dict name -> (N,) array, info dict)."""
    x, y, t, pen = traj.x, traj.y, traj.t, traj.pen
    n = len(x)
    down = pen > 0
    x0, x1 = x[down].min(), x[down].max()
    y0, y1 = y[down].min(), y[down].max()
    scale = max(x1 - x0, y1 - y0, 1e-9)          # one scale: aspect ratio is preserved
    xn, yn = (x - x0) / scale, (y - y0) / scale

    dx = np.diff(xn, prepend=xn[0])
    dy = np.diff(yn, prepend=yn[0])
    dt = np.diff(t, prepend=t[0])
    dt_safe = np.clip(dt, config.MIN_DT, config.MAX_DT)
    dist = np.hypot(dx, dy)
    speed = dist / dt_safe
    speed[0] = 0.0
    accel = np.diff(speed, prepend=speed[0]) / dt_safe

    moving = dist > 1e-9
    heading = np.arctan2(dy, dx)
    # carry the last defined heading over points that did not move
    idx = np.where(moving, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    heading_filled = np.where(moving[idx], heading[idx], 0.0)
    turn = np.diff(heading_filled, prepend=heading_filled[0])
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    turn[~moving] = 0.0

    sid, starts = stroke_ids(pen)
    seg = np.where(down, dist, 0.0)
    n_strokes = int(starts.sum())
    stroke_len = np.zeros(max(n_strokes, 1))
    np.add.at(stroke_len, np.clip(sid, 0, None)[down], seg[down])
    point_stroke_len = np.where(down & (sid >= 0), stroke_len[np.clip(sid, 0, None)], 0.0)

    az = np.deg2rad(traj.azimuth)
    feats = {
        "x": xn, "y": yn, "dx": dx, "dy": dy,
        "log_dt": np.log(dt_safe),
        "log_speed": np.log1p(speed),
        "log_accel": np.sign(accel) * np.log1p(np.abs(accel)),
        "sin_heading": np.where(moving, np.sin(heading), 0.0),
        "cos_heading": np.where(moving, np.cos(heading), 0.0),
        "turn": turn / np.pi,
        "pen": pen.astype(np.float64),
        "stroke_start": starts.astype(np.float64),
        "log_stroke_len": np.log1p(point_stroke_len),
        # optional channels stay NaN where not reported
        "pressure": traj.pressure.copy(),
        "altitude": traj.altitude / 90.0,
        "sin_azimuth": np.sin(az),
        "cos_azimuth": np.cos(az),
    }
    info = {"tokens": n, "strokes": n_strokes, "duration_s": round(float(t[-1] - t[0]), 2)}
    return feats, info


def optional_channel_available(values, pen):
    """A pen channel counts as available only if it varies over pen-down points.
    Mice report a constant 0.5 'pressure' and many touch screens a constant 0/1,
    which carry no information and must not be mistaken for real pressure."""
    v = values[pen > 0]
    v = v[np.isfinite(v)]
    return v.size >= max(5, 0.5 * (pen > 0).sum()) and float(np.ptp(v)) > 1e-3


def feature_matrix(traj, feature_names, mean=None, std=None):
    """-> (N, F) float32 matrix (standardised when mean/std given), list of defaulted features.

    A missing optional channel is filled with the training mean (0 after
    standardisation), i.e. "average pen behaviour", and reported back.
    """
    feats, info = compute_features(traj)
    defaulted = []
    channel_of = {"pressure": "pressure", "altitude": "altitude",
                  "sin_azimuth": "azimuth", "cos_azimuth": "azimuth"}
    cols = []
    for j, name in enumerate(feature_names):
        col = feats[name].astype(np.float64)
        if name in channel_of:
            available = optional_channel_available(getattr(traj, channel_of[name]), traj.pen)
            fill = mean[j] if mean is not None else 0.0
            if not available:
                col = np.full_like(col, fill)
                defaulted.append(name)
            else:
                # in-air points legitimately have no pressure/tilt on some devices
                col = np.where(np.isfinite(col), col, 0.0 if name == "pressure" else fill)
        cols.append(col)
    m = np.stack(cols, axis=1)
    if mean is not None:
        m = (m - mean) / std
        m = np.clip(m, -config.FEATURE_CLIP, config.FEATURE_CLIP)
    if not np.all(np.isfinite(m)):
        raise HandwritingInputError("the handwriting produced invalid numbers")
    info["defaulted_features"] = defaulted
    return m.astype(np.float32), info


def windows(n, max_len=config.MAX_LEN):
    """Start/end indices of overlapping windows (stride max_len/2) covering 0..n."""
    if n <= max_len:
        return [(0, n)]
    stride = max_len // 2
    starts = list(range(0, n - max_len + 1, stride))
    if starts[-1] + max_len < n:
        starts.append(n - max_len)
    return [(s, s + max_len) for s in starts]
