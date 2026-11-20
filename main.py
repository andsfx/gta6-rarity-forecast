"""gta6-trophy-rarity-forecast-01: a refresh-aware quantile forecast of GTA VI trophy earn rates.

The grader runs ``python3 -P -s /bot/main.py`` once per case, in a sandbox with no network. The
program reads every ``*.in`` file in ``$MAKEFUN_INPUT_DIR`` (default ``/input``) in name order and
writes ``<id>.out`` to ``$MAKEFUN_OUTPUT_DIR`` (default ``/output``):

    {"trophies": [{"id": 4, "q": [q05, q25, q50, q75, q95]}, ...]}

Every number is a percent: the quantiles of the trophy's earn rate at the case's target time, five
finite values in 0..100, non-decreasing, one row per trophy and one id each.

The model lives on the logit scale, where a rate x percent becomes ln(p / (1 - p)) of p = x / 100
after x is clamped to 0.01..99.99 (0% and 100% have no logit). The quantiles are

    100 / (1 + exp(-(mu + z_tau * sigma))),   z = [-1.645, -0.674, 0, 0.674, 1.645]

so mu is the centre on the logit scale and sigma the spread there. Two regimes pick mu.

Regime A, a stale page. Every read of the case shows the same rate for the trophy, so Exophase
has not refreshed yet and the reads only repeat one value. The next refresh dilutes the rate: it
adds many more tracked players than earners. The launch-night refresh multiplied the tracked
players by about six, and new earners offset only part of that, so the centre shifts down by
ALPHA * ln(K) logit units with K = 6 and ALPHA = 0.85.

Regime B, visible refreshes. The trophy shows at least two distinct rates, so a refresh happened
inside the history. Fitting logit(rate) against ln(players tracked) by least squares gives the
dilution law. Each trophy's slope is shrunk halfway toward the field's median slope: most of the
field dilutes, and single-trophy slopes are noisy. A second fit of ln(players) against read time
projects the players at the target; the projection is capped at thirty times the last read's
count, and the part of the projected growth beyond the largest single-refresh jump the history
has shown (dmax) counts only 0.4x, so a long horizon cannot run away on a fit whose data span is
shorter than the horizon. The spread comes from the shrunk fit's root mean square residual with a
0.35 floor, plus a term that grows with the square root of the horizon in days, capped at 1.6.

A trophy already at the rate ceiling (100.00%) has never diluted in the measured field, so it is
pinned there with a narrow spread instead of being shifted down.

Mu is clamped to logit(0.02)..logit(99.0) in Regime B, every quantile is clamped to
0.0001..99.9999, sorted, made non-decreasing and rounded to four decimals. A trophy with no rate
anywhere in the history gets a fixed wide row [0.1, 1.0, 5.0, 20.0, 60.0]. Every step is plain
arithmetic over the case file, so two runs write identical bytes.
"""

import json
import math
import os
import statistics
import sys

# The standard normal's quantile at each level, the order every row lists them in.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# The stale-page shift: mu = logit(r_last) - ALPHA * ln(K).
ALPHA = 0.85
K = 6.0
# Regime A's base spread; slightly wider than v1 because wider tails are cheap insurance.
SIGMA_A = 0.55
# A trophy at the rate ceiling is pinned there with this narrow spread.
PIN_SIGMA = 0.35

# Rates are clamped here before the logit, and quantiles into the wider bounds after it.
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
QUANTILE_FLOOR = 0.0001
QUANTILE_CEIL = 99.9999

# A trophy the history never lists falls back to this row.
FALLBACK_Q = (0.1, 1.0, 5.0, 20.0, 60.0)

DAY_MINUTES = 1440.0
# The launch-night read spacing, 90 minutes, is the unit of the horizon-scaled spread terms.
DAY_UNIT = 0.0625
# Regime B clamps the players projection to this multiple of the last read's count.
PLAYERS_MAX_GROWTH = 30.0
# Pooled slope shrinkage: b'_i = SHRINK * b_i + (1 - SHRINK) * b_pool.
SHRINK = 0.5
# The part of a projected dln beyond the largest observed single-refresh jump that counts.
DAMP_FACTOR = 0.4
# Regime B's spread: max(SIGMA_B_FLOOR, sd_residuals) + SIGMA_B_HORIZON * sqrt(horizon_days),
# capped at SIGMA_B_CAP.
SIGMA_B_FLOOR = 0.35
SIGMA_B_HORIZON = 0.10
SIGMA_B_CAP = 1.6


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


def history_max_dln(history):
    """The largest ln(players) jump between consecutive reads with distinct player counts.

    dmax: the biggest single-refresh growth the history has actually shown. Zero when the
    history holds no two consecutive distinct counts. Projected growth beyond dmax is damped.
    """
    largest = 0.0
    prev = None
    for read in history:
        if not isinstance(read, dict):
            continue
        players = finite_number(read.get("players_tracked"))
        if players is None or players <= 0.0:
            continue
        if prev is not None and players != prev:
            largest = max(largest, math.log(players) - math.log(prev))
        prev = players
    return largest


def regime_a(r_last, horizon_days):
    """A stale page: shift the last rate down by the launch-night dilution and widen with time."""
    mu = logit(r_last) - ALPHA * math.log(K)
    if horizon_days <= 1.0:
        sigma = SIGMA_A
    else:
        sigma = min(1.6, SIGMA_A * math.sqrt(horizon_days / DAY_UNIT))
    return mu, sigma


def regime_b(series, history, target, horizon_days, b_pool, dmax):
    """Visible refreshes: fit logit(rate) on ln(players) with a shrunk slope, project the
    players with damped extrapolation, and read the dilution line there."""
    fit_points = []
    logits = []
    for players, rate in series:
        lg = logit(rate)
        logits.append(lg)
        if players is not None:
            fit_points.append((math.log(players), lg))

    fit = linear_fit(fit_points)
    if fit is not None:
        # Shrink the noisy per-trophy slope halfway toward the field's median slope.
        slope = SHRINK * fit[0] + (1.0 - SHRINK) * b_pool
        # Re-anchor the shrunk line at the centroid of the fit points, so the line still
        # passes through the observed mean instead of drifting with the raw intercept.
        n = len(fit_points)
        mean_x = sum(point[0] for point in fit_points) / n
        mean_y = sum(point[1] for point in fit_points) / n
        intercept = mean_y - slope * mean_x
    else:
        slope = 0.0
        intercept = sum(logits) / len(logits) if logits else 0.0

    if len(fit_points) >= 3:
        rms = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in fit_points) / len(fit_points)
        )
        sd = max(SIGMA_B_FLOOR, rms)
    else:
        sd = 0.45
    sigma = min(SIGMA_B_CAP, sd + SIGMA_B_HORIZON * math.sqrt(max(0.0, horizon_days)))

    # The players at the target: ln(players) against read time over the whole case history.
    last_players = None
    for players, _rate in reversed(series):
        if players is not None:
            last_players = players
            break
    if last_players is None:
        for read in reversed(history):
            if not isinstance(read, dict):
                continue
            players = finite_number(read.get("players_tracked"))
            if players is not None and players > 0.0:
                last_players = players
                break

    target_players = last_players
    if last_players is not None and target is not None:
        timed = []
        origin = None
        for read in history:
            if not isinstance(read, dict):
                continue
            players = finite_number(read.get("players_tracked"))
            if players is None or players <= 0.0:
                continue
            instant = parse_time(read.get("at"))
            if instant is None:
                instant = parse_time(read.get("slot"))
            if instant is None:
                continue
            if origin is None:
                origin = instant
            timed.append(((instant - origin) / 86400.0, math.log(players)))
        growth = linear_fit(timed)
        if growth is not None:
            slope_t, intercept_t = growth
            try:
                projected = math.exp(intercept_t + slope_t * ((target - origin) / 86400.0))
            except OverflowError:
                projected = float("inf")
            if math.isfinite(projected) and projected > 0.0:
                target_players = projected
        if target_players is None or not math.isfinite(target_players):
            target_players = last_players
        target_players = clamp(
            target_players, last_players, last_players * PLAYERS_MAX_GROWTH
        )
        # Damp the extrapolated growth: the part of dln beyond the largest single-refresh
        # jump the history has shown counts only DAMP_FACTOR. The clamp after keeps the
        # projection inside the guards for every edge case.
        dln = math.log(target_players) - math.log(last_players)
        if dln > dmax:
            dln = dmax + DAMP_FACTOR * (dln - dmax)
            target_players = last_players * math.exp(dln)
            target_players = clamp(
                target_players, last_players, last_players * PLAYERS_MAX_GROWTH
            )

    if last_players is not None and target_players is not None:
        mu = intercept + slope * math.log(target_players)
    else:
        mu = intercept
    mu = clamp(mu, logit(0.02), logit(99.0))
    return mu, sigma


def forecast_trophy(series, history, target, horizon_days, b_pool, dmax):
    """The five quantiles of one trophy's rate at the target, in percent, rounded to 4 decimals."""
    if not series:
        return list(FALLBACK_Q)

    rate_values = [rate for _players, rate in series]
    r_last = rate_values[-1]

    if r_last >= RATE_CEIL - 1e-9:
        # A 100.00% trophy has never diluted in the measured field: pin it at the ceiling
        # with a narrow spread instead of shifting it down.
        mu = logit(RATE_CEIL)
        sigma = PIN_SIGMA
    elif len(set(rate_values)) <= 1:
        mu, sigma = regime_a(r_last, horizon_days)
    else:
        mu, sigma = regime_b(series, history, target, horizon_days, b_pool, dmax)

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
    target = parse_time(case.get("target"))

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

    # The field's pooled dilution slope: the median per-trophy slope over every regime B
    # trophy whose fit exists. Noisy single-trophy slopes shrink halfway toward it.
    slopes = []
    for _trophy_id, series in prepared:
        rate_values = [rate for _players, rate in series]
        if len(set(rate_values)) <= 1:
            continue
        fit_points = [
            (math.log(players), logit(rate)) for players, rate in series if players is not None
        ]
        fit = linear_fit(fit_points)
        if fit is not None:
            slopes.append(fit[0])
    b_pool = statistics.median(slopes) if slopes else 0.0
    dmax = history_max_dln(history)

    rows = []
    for trophy_id, series in prepared:
        rows.append(
            {
                "id": trophy_id,
                "q": forecast_trophy(series, history, target, horizon_days, b_pool, dmax),
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
