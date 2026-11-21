"""gta6-trophy-rarity-forecast-01: v3, a hierarchical-strata quantile forecast of GTA VI
trophy earn rates.

The grader runs ``python3 -P -s /bot/main.py`` once per case, in a sandbox with no network. The
program reads every ``*.in`` file in ``$MAKEFUN_INPUT_DIR`` (default ``/input``) in name order and
writes ``<id>.out`` to ``$MAKEFUN_OUTPUT_DIR`` (default ``/output``):

    {"trophies": [{"id": 4, "q": [q05, q25, q50, q75, q95]}, ...]}

Every number is a percent: the quantiles of the trophy's earn rate at the case's target time, five
finite values in 0..100, non-decreasing, one row per trophy and one id each.

The model lives on the logit scale, where a rate x percent becomes ln(p / (1 - p)) of p = x / 100
after x is clamped to 0.01..99.99 (0% and 100% have no logit). The quantiles are

    100 / (1 + exp(-(mu + z_tau * sigma))),   z = [-1.645, -0.674, 0, 0.674, 1.645]

so mu is the centre on the logit scale and sigma the spread there.

Step 1, the dilution slope. A trophy with at least two distinct rates across its history gets a
least squares line through logit(rate) against ln(players tracked), whose slope b_i is the
per-trophy dilution law. A trophy with fewer than two distinct rates sits on a stale page - the
reads only repeat one value, and the next refresh dilutes it. That branch shifts the centre down
by 0.85 * ln(6) logit units (launch night multiplied the tracked players by about six) below the
last rate and stops there, with a flat spread of 0.55.

Step 2, the strata. The last rate cuts the field into S0 r >= 15, S1 2 <= r < 15, S2 0.2 <= r < 2
and S3 r < 0.2.

Step 3, hierarchical shrinkage. b_strata is the median of the finite b_j of the trophies in the
same strata, and the working slope is b_shrunk = (b_i + 2 * b_strata) / 3: a median built from
dozens of trophies is far more stable than one trophy's line.

Step 4, power-law players. ln(N_t) = a + b * ln(t + c) is fit over all reads, t the days since the
first read, with c searched over a grid and each fit scored by its SSE on the last eight reads
only, so the law that best tracks the most recent growth wins. Reading the winner at the case's
target time gives ln(N_target); the growth dlnN = ln(N_target) - ln(N_last) is held between 1x and
50x the last count.

Step 5, the centre. mu = logit(r) + b_shrunk * dlnN + drift_per_day * horizon_days, where the
drift is a cross-game estimate of how each strata's median moves per day as players work through
the story (S1 story trophies rebound, S2 and S3 follow weaker, S0 none).

Step 6, the spread. sigma_samp = 0.2 * sqrt(1 / (N p (1-p))) is the sampling noise of the last
count itself, kept small so it stays a floor. The residual part is the root mean square of the
(logit(rate) - fitted line) residuals when the history has four or more distinct rates, or a
per-strata table otherwise. sigma = max(0.20, min(1.60, sigma_samp + sd_resid)) times
(1 + 0.10 * sqrt(horizon_days)), capped at 1.60 overall.

Step 7, the quantiles. inv_logit(mu + z * sigma) at the five z levels, rounded to four decimals
and made non-decreasing. A trophy whose last rate is 100.00% is pinned at 99.99 on every level.
A trophy with no rate anywhere in the history gets a fixed wide row [0.1, 1.0, 5.0, 20.0, 60.0].
Every step is plain arithmetic over the case file, so two runs write identical bytes.
"""

import json
import math
import os
import statistics
import sys

# The standard normal's quantile at each level, the order every row lists them in.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Rates are clamped here before the logit, and quantiles into the wider bounds after it.
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
QUANTILE_FLOOR = 0.0001
QUANTILE_CEIL = 99.9999

# A trophy the history never lists with a rate falls back to this row.
FALLBACK_Q = (0.1, 1.0, 5.0, 20.0, 60.0)

# A trophy whose last rate is exactly the page's 100.00 ceiling is pinned here, every level.
PIN_Q = 99.99
PIN_THRESHOLD = 100.0 - 1e-9

DAY_MINUTES = 1440.0

# The stale page: mu = logit(r_last) + STALE_SHIFT, flat spread, and no horizon widening.
STALE_SHIFT = -0.85 * math.log(6.0)
STALE_SIGMA = 0.55

# Per-strata spread table for trophies whose history is too short for residuals.
STRATA_SIGMA = (0.55, 0.50, 0.35, 0.45)
# Cross-game estimate of how the strata median moves per day, in logit units.
STRATA_DRIFT_PER_DAY = (0.00, 0.92 / 4.5, 0.46 / 4.5, 0.10 / 4.5)

# The power-law player projection: the c grid, the SSE scored on the last SSE_TAIL reads only,
# and the projected ratio of players at the target to the last count held inside [1, RATIO_MAX].
C_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
SSE_TAIL = 8
RATIO_MAX = 50.0

# Regime B's spread: max(0.20, min(1.60, SAMP_SCALE * sqrt(1 / (N p (1-p))) + sd_resid)) times
# (1 + SIGMA_HORIZON * sqrt(horizon_days)), capped at SIGMA_CAP overall.
SAMP_SCALE = 0.2
P_FLOOR = 0.0001
P_CEIL = 0.9999
SIGMA_FLOOR = 0.20
SIGMA_CAP = 1.60
SIGMA_HORIZON = 0.10


def clamp(x, lo, hi):
    """x held inside [lo, hi]; NaN cannot pass through the arithmetic below."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def logit(pct):
    """Logit of a rate in percent, clamped to 0.01..99.99 so 0% and 100% stay finite."""
    p = clamp(pct, RATE_FLOOR, RATE_CEIL) / 100.0
    return math.log(p / (1.0 - p))


def percent(z):
    """Inverse logit: 100 / (1 + exp(-z)), written so a large |z| cannot overflow exp()."""
    if z >= 0.0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x to four decimals, half up, as the house's JavaScript Math.round(x * 10000) / 10000.

    A whole number comes back as an int so json writes 100 where JavaScript writes 100.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def days_from_civil(year, month, day):
    """Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm)."""
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_time(text):
    """ISO 8601 UTC time as seconds since the epoch, or None when it does not read.

    The case files use ``2026-11-19T03:00:00Z`` and ``2026-11-19T03:00:00.309Z``. A trailing
    ``Z`` or a numeric offset is accepted; the offset is applied when present.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        date_part, sep, time_part = s.partition("T")
        if not sep:
            date_part, sep, time_part = s.partition(" ")
            if not sep:
                return None
        year, month, day = (int(piece) for piece in date_part.split("-"))
        offset = 0
        if time_part.endswith("Z") or time_part.endswith("z"):
            time_part = time_part[:-1]
        for sign in ("+", "-"):
            cut = time_part.rfind(sign)
            if cut > 0:
                zone = time_part[cut + 1:]
                time_part = time_part[:cut]
                hours, _, minutes = zone.partition(":")
                step = int(hours) * 3600 + int(minutes or "0") * 60
                offset = step if sign == "+" else -step
                break
        pieces = time_part.split(":")
        if len(pieces) < 3:
            return None
        hours = int(pieces[0])
        minutes = int(pieces[1])
        seconds = float(pieces[2])
        return (
            days_from_civil(year, month, day) * 86400.0
            + hours * 3600.0
            + minutes * 60.0
            + seconds
            - offset
        )
    except (ValueError, TypeError):
        return None


def finite_number(value):
    """value as a float when it is a real number JSON can carry, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def linear_fit(points):
    """Least squares (slope, intercept) of y on x over [(x, y), ...], or None if x never varies."""
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    var_x = sum((p[0] - mean_x) ** 2 for p in points) / n
    if var_x <= 1e-15:
        return None
    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / n
    slope = cov / var_x
    return slope, mean_y - slope * mean_x


def trophy_series(history, index):
    """[(players or None, rate), ...] over the reads that list the trophy, oldest first."""
    series = []
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list) or index >= len(rates):
            continue
        rate = finite_number(rates[index])
        if rate is None:
            continue
        players = finite_number(read.get("players_tracked"))
        if players is not None and players <= 0.0:
            players = None
        series.append((players, rate))
    return series


def read_time(read):
    """A read's instant in epoch seconds: its at field, then its slot, or None."""
    instant = parse_time(read.get("at"))
    if instant is None:
        instant = parse_time(read.get("slot"))
    return instant


def stratum_of(rate):
    """S0 if r >= 15, S1 if 2 <= r < 15, S2 if 0.2 <= r < 2, S3 if r < 0.2."""
    if rate >= 15.0:
        return 0
    if rate >= 2.0:
        return 1
    if rate >= 0.2:
        return 2
    return 3


def dilution_fit(series):
    """Least squares of logit(rate) on ln(players).

    Returns (fit_points, (slope, intercept)) where fit_points are the [(ln N, logit rate), ...]
    the line was fit on, or (fit_points, None) when the line is not identified (ln N never
    varies, or fewer than two points carry a player count).
    """
    fit_points = []
    for players, rate in series:
        if players is not None:
            fit_points.append((math.log(players), logit(rate)))
    return fit_points, linear_fit(fit_points)


def players_projection(history, target, horizon_days):
    """ln(players) at the target from the best power law over the reads, or None.

    Fits ln(N) = intercept + slope * ln(t + c) with t the days since the first read. Each c of
    the grid gets a closed-form least squares fit, scored by its SSE on the last reads only, and
    the winner is read at the target's time.
    """
    timed = []
    origin = None
    for read in history:
        if not isinstance(read, dict):
            continue
        players = finite_number(read.get("players_tracked"))
        if players is None or players <= 0.0:
            continue
        instant = read_time(read)
        if instant is None:
            continue
        base = origin if origin is not None else instant
        origin = base
        timed.append(((instant - base) / 86400.0, math.log(players)))
    if len(timed) < 2:
        return None

    target_time = parse_time(target) if isinstance(target, str) else None
    if target_time is not None:
        t_target = max(0.0, (target_time - origin) / 86400.0)
    else:
        t_target = timed[-1][0] + horizon_days

    best = None  # (sse, intercept, slope, c)
    for c in C_GRID:
        xs = []
        ys = []
        usable = True
        for t, y in timed:
            arg = t + c
            if arg <= 0.0:
                usable = False
                break
            xs.append(math.log(arg))
            ys.append(y)
        if not usable:
            continue
        fit = linear_fit(list(zip(xs, ys)))
        if fit is None:
            continue
        slope, intercept = fit
        tail = min(SSE_TAIL, len(xs))
        sse = 0.0
        for k in range(len(xs) - tail, len(xs)):
            diff = ys[k] - (intercept + slope * xs[k])
            sse += diff * diff
        if best is None or sse < best[0]:
            best = (sse, intercept, slope, c)
    if best is None:
        return None

    _sse, intercept, slope, c = best
    arg = t_target + c
    if arg <= 0.0:
        return None
    return intercept + slope * math.log(arg)


def last_players(series, history):
    """The trophy's last tracked-player count, from its series, then from the history."""
    for players, _rate in reversed(series):
        if players is not None:
            return players
    for read in reversed(history):
        if not isinstance(read, dict):
            continue
        players = finite_number(read.get("players_tracked"))
        if players is not None and players > 0.0:
            return players
    return None


def emit_quantiles(mu, sigma):
    """The five rounded, non-decreasing percent quantiles of inv_logit(mu + z * sigma)."""
    if not (math.isfinite(mu) and math.isfinite(sigma)):
        return list(FALLBACK_Q)
    raw = [percent(mu + z * sigma) for z in Z]
    if any(not math.isfinite(value) for value in raw):
        return list(FALLBACK_Q)

    quantiles = [clamp(value, QUANTILE_FLOOR, QUANTILE_CEIL) for value in raw]
    quantiles.sort()
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    rounded = [round4(q) for q in quantiles]
    for i in range(1, len(rounded)):
        if rounded[i] < rounded[i - 1]:
            rounded[i] = rounded[i - 1]
    return rounded


def forecast_trophy(entry, history, horizon_days, ln_target, strata_slopes):
    """The five quantiles of one trophy's rate at the target, in percent, rounded to 4 decimals."""
    series = entry["series"]
    if not series:
        return list(FALLBACK_Q)

    r_last = entry["r_last"]
    if r_last >= PIN_THRESHOLD:
        # A 100.00% trophy has never diluted in the measured field: pin every level.
        return [PIN_Q, PIN_Q, PIN_Q, PIN_Q, PIN_Q]

    if entry["distinct"] < 2:
        # Stale page: shift the last rate down by the launch-night dilution and stop.
        return emit_quantiles(logit(r_last) + STALE_SHIFT, STALE_SIGMA)

    stratum = entry["stratum"]
    fit = entry.get("fit")
    fit_points = entry.get("fit_points", [])

    # Step 3: shrink the noisy per-trophy slope two-thirds toward its strata's median slope.
    if fit is not None:
        slope_i, intercept_i = fit
        b_shrunk = (slope_i + 2.0 * strata_slopes[stratum]) / 3.0
    else:
        slope_i = intercept_i = None
        b_shrunk = strata_slopes[stratum]

    # Step 4: the projected growth, in ln(players), held between 1x and 50x the last count.
    n_last = last_players(series, history)
    if ln_target is not None and n_last is not None:
        dlnN = clamp(ln_target - math.log(n_last), 0.0, math.log(RATIO_MAX))
    else:
        dlnN = 0.0

    # Step 5: the centre.
    mu = logit(r_last) + b_shrunk * dlnN + STRATA_DRIFT_PER_DAY[stratum] * horizon_days

    # Step 6: the spread - the sampling noise of the last count plus the fit's residual noise.
    p = clamp(r_last / 100.0, P_FLOOR, P_CEIL)
    if n_last is not None:
        sigma_samp = SAMP_SCALE * math.sqrt(1.0 / (n_last * p * (1.0 - p)))
    else:
        sigma_samp = 0.0
    if fit is not None and entry["distinct"] >= 4 and fit_points:
        squares = 0.0
        for x, y in fit_points:
            diff = y - (intercept_i + slope_i * x)
            squares += diff * diff
        sd_resid = math.sqrt(squares / len(fit_points))
    else:
        sd_resid = STRATA_SIGMA[stratum]
    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma_samp + sd_resid))
    sigma *= 1.0 + SIGMA_HORIZON * math.sqrt(max(0.0, horizon_days))
    sigma = min(SIGMA_CAP, sigma)
    return emit_quantiles(mu, sigma)


def forecast_case(case):
    """The output document for one parsed case."""
    if not isinstance(case, dict):
        raise ValueError("the case is not a JSON object")
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        raise ValueError("the case lists no trophies")

    history = case.get("history")
    if not isinstance(history, list):
        history = []

    horizon_minutes = finite_number(case.get("horizon_minutes"))
    if horizon_minutes is None or horizon_minutes < 0.0:
        horizon_minutes = 0.0
    horizon_days = horizon_minutes / DAY_MINUTES
    target = case.get("target")

    prepared = []
    seen = set()
    for index, trophy in enumerate(trophies):
        if isinstance(trophy, dict):
            trophy_id = trophy.get("id")
        else:
            trophy_id = None
        if isinstance(trophy_id, bool) or not isinstance(trophy_id, (int, float)):
            trophy_id = index + 1
        elif isinstance(trophy_id, float) and trophy_id.is_integer():
            trophy_id = int(trophy_id)
        if trophy_id in seen:
            continue
        seen.add(trophy_id)
        prepared.append((trophy_id, trophy_series(history, index)))

    # Each trophy's rate profile, and per-strata pools of finite dilution slopes.
    entries = []
    pools = [[], [], [], []]
    for trophy_id, series in prepared:
        entry = {"id": trophy_id, "series": series}
        if series:
            rate_values = [rate for _players, rate in series]
            r_last = rate_values[-1]
            entry["r_last"] = r_last
            entry["distinct"] = len(set(rate_values))
            entry["stratum"] = stratum_of(r_last)
            if entry["distinct"] >= 2 and r_last < PIN_THRESHOLD:
                fit_points, fit = dilution_fit(series)
                entry["fit_points"] = fit_points
                entry["fit"] = fit
                if fit is not None:
                    slope_i, _intercept_i = fit
                    if math.isfinite(slope_i):
                        pools[entry["stratum"]].append(slope_i)
        entries.append(entry)

    strata_slopes = {}
    for s in range(4):
        strata_slopes[s] = statistics.median(pools[s]) if pools[s] else 0.0

    # The player projection is the whole page's history, so it is fit once per case.
    ln_target = players_projection(history, target, horizon_days)

    rows = []
    for entry in entries:
        rows.append(
            {
                "id": entry["id"],
                "q": forecast_trophy(entry, history, horizon_days, ln_target, strata_slopes),
            }
        )
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    except OSError as error:
        sys.stderr.write("gta6 forecast: cannot list %s: %s\n" % (input_dir, error))
        return

    for name in names:
        path = os.path.join(input_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            document = forecast_case(case)
            payload = json.dumps(document, separators=(",", ":"), ensure_ascii=True) + "\n"
        except Exception as error:  # one bad case must not cost the others their outputs
            sys.stderr.write("gta6 forecast: %s: %s\n" % (name, error))
            continue
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
