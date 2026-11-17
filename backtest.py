"""Leave-one-out backtest of this entry's forecaster on the kit's comparable launches.

Every comparable a case ships is a past launch with a full release day, so each one can be
forecast as if it were the target: its first hour is the print, the other launches are the
training set, and its real hours 2..24 are the answer. The score is the contest's own loss
(the mean pinball loss on the log scale, scorer/forecast.ts's formula as the rules state
it), computed for this entry's model and for the house persistence baseline on the same
launches, so the two are comparable number for number.

Usage: python3 backtest.py <path to a kit case file, e.g. the kit's input/p00.in>

The case file is the kit's, not this repository's: the kit asks entrants not to
bulk-download its source (SullyGnome) again, and nothing here needs more than the one
file. The Grand Theft Auto VI pre-release curve is excluded, as in main.py: it is not a
launch and has no full release day.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as entry

LEVELS = entry.LEVELS
HOURS = list(range(2, 25))


def pinball(rows, day):
    """The contest's loss of one forecast: mean over hours 2..24 and the five levels."""
    total = 0.0
    count = 0
    for row in rows:
        y = max(1.0, day[row["hour"] - 1])
        for tau, q in zip(LEVELS, row["q"]):
            d = math.log(y) - math.log(max(q, 1e-9))
            total += 2 * max(tau * d, (tau - 1) * d)
            count += 1
    return total / count


def main():
    case = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    launches = entry.training_launches(case)
    print(f"{len(launches)} comparable launches with a full release day")
    results = []
    for i, (first, day) in enumerate(launches):
        train = [launch for j, launch in enumerate(launches) if j != i]
        rows = [
            {"hour": h, "q": entry.model_quantiles(train, first, HOURS)[h]} for h in HOURS
        ]
        base = [{"hour": h, "q": entry.persistence_quantiles(first, h)} for h in HOURS]
        results.append(
            (first, max(day), pinball(rows, day), pinball(base, day))
        )
    for name, subset in [
        ("all launches", results),
        ("already live at hour 1 (first hour >= 20% of the day's peak)", [r for r in results if r[0] / r[1] >= 0.2]),
        ("first hour >= 100k viewers", [r for r in results if r[0] >= 100000]),
    ]:
        if not subset:
            continue
        model = sum(r[2] for r in subset) / len(subset)
        base = sum(r[3] for r in subset) / len(subset)
        print(f"{name}: n={len(subset)}  model loss {model:.4f}  persistence loss {base:.4f}")


if __name__ == "__main__":
    main()
