# gta6-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/andsfx/gta6-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/andsfx/gta6-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

This program forecasts the earn rate of every Grand Theft Auto VI trophy on Exophase at the case's target time, as five quantiles per trophy: the 5%, 25%, 50%, 75% and 95% levels. A rate is the percent of the players Exophase tracks for the game who have earned the trophy. The scorer compares the quantiles to the rate the house records at the target, so the forecast has to be both close and honest about how close it can be.

Everything happens on the logit scale, where a rate moves as ln(p / (1 - p)) and a miss from 1% to 2% costs about what a miss from 33% to 50% costs. That keeps rare trophies worth as much as common ones. The centre of the five quantiles comes from one of two regimes, and the spread comes from the horizon and, when a refresh is visible, from the residuals of the model's own fit.

The first regime is a stale page. Every read of the case shows the same rate for a trophy, which means Exophase has not refreshed yet and the next refresh will move it, mostly down. A refresh adds many more tracked players than new earners, so the rate dilutes. Launch night multiplied the tracked players by about six, and new earners offset only part of that, so the centre drops by 0.75 times ln(6) logit units below the last rate. The spread widens with the square root of the horizon.

The second regime is a page that has refreshed inside the history, so a trophy shows at least two distinct rates. A least squares line through logit(rate) against ln(players tracked) gives that trophy's dilution law, and a second line through ln(players) against read time projects the players at the target, held between one and fifty times the last read's count. The first line is then read at the projected count. The spread is the fit's root mean square residual plus a term that grows with the square root of the horizon, so a noisy history produces a wider forecast than a clean one.

Every quantile is clamped into 0.0001 to 99.9999, sorted, made non-decreasing and rounded to four decimals. A trophy the history never lists gets a fixed wide row. The program reads only its input file, uses the standard library, and does the same arithmetic twice, so two runs write identical bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=andsfx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
