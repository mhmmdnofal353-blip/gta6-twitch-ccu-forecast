"""gta6-twitch-ccu-forecast-01 entry: an analog (comparable-launch) quantile forecaster.

The idea, in one paragraph: a big game launch's release-day Twitch curve, divided by its own
first hour, has a shape that repeats across launches -- a launch that is already live around
the world at 00:00 UTC (a rolling midnight-local console unlock, like Grand Theft Auto VI)
ramps through the US unlocks and the US day, while a launch whose single global unlock falls
later in the UTC day explodes from a near-zero first hour. So this program takes every past
launch the case ships in `comparables`, reduces it to the 23 log-ratios ln(viewers_h /
viewers_1) of its release day, weights each launch by how close its first hour is to this
case's first hour (a Gaussian kernel on log first-hour viewers: launches of a similar size
behave alike, and the kernel quietly separates the two regimes above), and reads the five
scored quantiles of those weighted log-ratios, hour by hour. The forecast is hour 1's print
times those ratio quantiles.

Why this scores: the contest's loss is the pinball loss on the log scale, whose optimal
forecast is exactly the conditional quantiles of ln(viewers_h) given what is known -- and
ln(viewers_h) = ln(print_1) + ln(ratio_h), so empirical quantiles of the log-ratios of
comparable launches are the direct estimator. The house baseline instead fixes the median
ratio at 1 for every hour; real launches' median ratio is not 1 (it dips, then climbs well
above 1 into the US day), which is where this entry gains.

Leave-one-out backtest on the 37 comparable launches the kit ships (each launch forecast
from the other 36, scored with the contest's exact loss): on the 24 launches that, like
all five real targets, are already live at hour 1, this model scores 0.2075 against the
persistence baseline's 0.2652; over all 37 launches, 0.4088 against 0.9297. See
backtest.py and README.md.

The program is deterministic, standard library only, and never touches the network: for
each case it reads only that case's `comparables`, `known` and `forecast` fields. If the
comparables are unusable it falls back to the house persistence curve, so a forecast file
is always written and always valid.

Run locally the way the kit describes:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs it as `python3 -P -s /bot/main.py`; the sys.path line below puts this
file's own directory back on the path, as the kit's starter notes explain.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside
# this file would import (this entry is a single file, but the habit is the kit's own).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the fallback curve's shape).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The fallback (house persistence) spread, in log units per square-root hour.
FALLBACK_SIGMA = 0.15
# The kernel bandwidth on log first-hour viewers: two launches whose first hours differ by
# a factor of e^1 weigh about 0.61 of an exact match. Chosen on the backtest, where every
# bandwidth from 0.7 to 1.2 scored within noise of each other (about 0.208-0.211).
KERNEL_BANDWIDTH = 1.0
# A small shrink of the spread around the median, and a light smoothing of each quantile
# curve across neighbouring hours: with a few dozen launches the raw 5% and 95% quantiles
# are the noisiest numbers in the model, and both adjustments earned a small, consistent
# gain in the leave-one-out backtest (0.2091 for the raw quantiles, about 0.208 with them).
SPREAD_SCALE = 0.95
SMOOTHING = 0.3
# Fewer usable comparable launches than this and the empirical quantiles are not trusted.
MIN_LAUNCHES = 5


def parse_time(text):
    """An ISO 8601 UTC timestamp from a case file, as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int for compact JSON."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def weighted_quantile(pairs, tau):
    """The tau quantile of (value, weight) pairs, linear in the cumulative weight.

    The pairs are sorted by value; the quantile is read off the cumulative weight at
    tau * total, interpolating between the two values that straddle it.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        acc += weight
        if acc >= target:
            if i == 0:
                return value
            prev = ordered[i - 1][0]
            frac = (target - (acc - weight)) / weight if weight > 0 else 0.0
            return prev + (value - prev) * frac
    return ordered[-1][0]


def launch_day(comparable):
    """A comparable's 24 release-day hourly viewer counts, or None if it has no full day.

    The day runs from the comparable's own `day_start`; every hour must be present and
    positive for the launch to teach the model anything. This is what leaves out the one
    comparable that is not a launch at all (the GTA VI category's pre-release history,
    whose "day" is a trailer premiere with most hours missing).
    """
    try:
        start = parse_time(comparable["day_start"])
        by_hour = {}
        for point in comparable["hourly"]:
            by_hour[parse_time(point["from"])] = point["viewers"]
        day = []
        for h in range(24):
            viewers = by_hour.get(start + timedelta(hours=h))
            if viewers is None or viewers <= 0:
                return None
            day.append(float(viewers))
        return day
    except (KeyError, TypeError, ValueError):
        return None


def training_launches(case):
    """Every comparable of the case that has a full release day, as (first hour, day)."""
    launches = []
    for comparable in case.get("comparables") or []:
        day = launch_day(comparable)
        if day is not None:
            launches.append((day[0], day))
    return launches


def persistence_quantiles(p1, h):
    """The house baseline's row for hour h: print_1 with a widening log-normal spread."""
    return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]


def model_quantiles(launches, p1, hours):
    """The analog forecast: per hour, weighted quantiles of the launches' log-ratios.

    Three steps, all in log-ratio space: the weighted quantiles themselves; the spread
    shrink (each quantile moved SPREAD_SCALE of the way toward the median); and the
    smoothing (each hour's curve blended with its neighbours'), which runs last so the
    shrink is applied to the curves actually scored.
    """
    log_p1 = math.log(p1)
    weights = [
        math.exp(-0.5 * ((math.log(first) - log_p1) / KERNEL_BANDWIDTH) ** 2)
        for first, _ in launches
    ]
    raw = {}
    for h in hours:
        pairs = [
            (math.log(day[h - 1] / first), weight)
            for (first, day), weight in zip(launches, weights)
        ]
        ratios = [weighted_quantile(pairs, tau) for tau in LEVELS]
        median = ratios[2]
        raw[h] = [median + (r - median) * SPREAD_SCALE for r in ratios]
    rows = {}
    for h in hours:
        curve = []
        for j in range(len(LEVELS)):
            neighbours = [raw[k][j] for k in (h - 1, h + 1) if k in raw]
            if neighbours:
                value = (1 - SMOOTHING) * raw[h][j] + SMOOTHING * (
                    sum(neighbours) / len(neighbours)
                )
            else:
                value = raw[h][j]
            curve.append(round2(min(1e9, max(0.01, p1 * math.exp(value)))))
        rows[h] = sorted(curve)
    return rows


def forecast_case(case):
    """One case's forecast object: {"hours": [{"hour": h, "q": [...]}, ...]} for h in 2..24."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    launches = training_launches(case)
    if len(launches) >= MIN_LAUNCHES:
        rows = model_quantiles(launches, p1, hours)
    else:
        rows = {h: persistence_quantiles(p1, h) for h in hours}
    return {"hours": [{"hour": h, "q": rows[h]} for h in sorted(hours)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never fail a case: a crude but valid forecast beats an invalid one.
            forecast = {
                "hours": [
                    {"hour": h, "q": persistence_quantiles(1.0, h)} for h in range(2, 25)
                ]
            }
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
