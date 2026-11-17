# GTA VI Twitch CCU forecast — an analog quantile forecaster

An entry for make.fun's **gta6-twitch-ccu-forecast-01**: forecast, for each of hours 2–24
of Grand Theft Auto VI's release day (2026-11-19 UTC), five quantiles (5/25/50/75/95%) of
Twitch's concurrent viewers in the Grand Theft Auto VI category, from everything recorded
up to the end of hour 1. Scored by pinball loss on the log scale; a prize requires beating
both house entries, including the persistence baseline.

## The model

`main.py` (Python 3.13, standard library only) is an **analog forecaster** built on the
comparable launches every case ships:

1. Each comparable launch is reduced to its release-day **log-ratio curve**
   `ln(viewers_h / viewers_1)` for h = 2..24. (A comparable without a full, positive
   release day is dropped — this excludes the one non-launch in the set, the GTA VI
   category's own pre-release history.)
2. Each launch is weighted by a **Gaussian kernel on log first-hour viewers**
   (bandwidth 1.0): launches whose first hour is close to this case's first hour count
   most. First-hour size is the regime signal — a rolling midnight-local console launch
   (GTA VI's pattern, and every practice target's) is already live worldwide at 00:00 UTC,
   while a single global unlock later in the UTC day starts near zero and explodes; the
   kernel keeps the two from contaminating each other without any hand-made switch.
3. For each hour, the forecast's five quantiles are the **weighted empirical quantiles**
   of the log-ratios, with a small spread shrink (×0.95 toward the median) and light
   smoothing across neighbouring hours (0.3), then exponentiated back onto hour 1's print.

Because the loss is pinball on the log scale and `ln(viewers_h) = ln(print_1) +
ln(ratio_h)`, empirical quantiles of comparable log-ratios are the direct estimator of
what the scorer rewards. The house baseline fixes every hour's median ratio at 1; real
launches dip, then climb well above 1 into the US day — that shape is the edge.

If a case ever ships too few usable comparables, the program falls back to the house
persistence curve, so its output is always written and always valid. It is deterministic
(two runs are byte-identical), offline, and runs all four practice cases in ~0.25 s of
the 8 s per-case cap.

## Backtest

`backtest.py` replays every comparable launch as the target (leave-one-out: forecast from
the other 36, scored with the contest's exact loss, using the model's own functions):

| targets | n | this model | persistence baseline |
| --- | --- | --- | --- |
| all comparable launches | 37 | **0.4088** | 0.9297 |
| already live at hour 1 (≥20% of day peak — the regime of all five real targets) | 24 | **0.2075** | 0.2652 |
| first hour ≥ 100k viewers | 16 | **0.2142** | 0.2611 |

Run it with a kit case file: `python3 backtest.py /path/to/kit/input/p00.in`.
(The practice cases' own answers are not in the kit, so the comparables are the honest
validation set; the house baseline's published practice loss is 0.387.)

## Running

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

reads every `<id>.in` in the input directory and writes `<id>.out` beside the requested
output directory, in the contest's format: `{"hours": [{"hour": 2, "q": [q05, q25, q50,
q75, q95]}, …]}`, 23 rows, hours 2–24.

## Layout

- `main.py` — the entry (the grader runs `python3 -P -s /bot/main.py`).
- `backtest.py` — the leave-one-out validation above.
- `LICENSE` — MIT (the kit starter's licence).
- `.makefun/owner.json` — make.fun ownership proof for the entrant's payout wallet.

Built by Muse (Muse Spark) working for Kang (@keen-coyote-41) in recorded agent
sessions; referral: https://make.fun/r/h7qfy6r1
