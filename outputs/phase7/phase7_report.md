# Phase 7: data-centre siting screen (model predictions + solver verification)

> This is an approximate open-data model, **not AESO's grid**. Costs are illustrative. Results show the *method* and do not give real
> interconnection capacity. Real siting needs AESO connection studies.

## Question
At which 138/240 kV buses could a new data centre of 100, 300 or 500 MW (pf 0.98) connect without violating limits at peak load?
How much would it raise the system dispatch cost?

## Method (`src/abgrid/usecase/siting_screen.py`, config block `siting`)
1. **Base snapshots.** Two snapshots are built from 2024 metered data, using the same structure as scenario run `s1250c` with zero
   noise. Each is solved with AC-OPF at nominal limits.
   - Winter peak: 2024-01-25 18:00. Cost 262,516 $/h, median LMP 59 $/MWh.
   - Summer peak: 2024-07-22 17:00 (the July hour with the largest planning-area load). Cost 332,221 $/h, median LMP 106 $/MWh.
2. **Screen.** There are 819 candidate buses (138 and 240 kV, boundary buses excluded). Across 3 sizes and 2 snapshots this gives
   **4,914 cases**. Each case adds the load to the graph, and the refined GridSFM checkpoint (Phase 6b) predicts:
   - P(feasible), i.e. relaxation level 0;
   - the dispatch cost;
   - the bus voltages.

   The predicted Δcost is the candidate's cost minus the *predicted* base cost. This cancels most of the model's own bias.
3. **Verify.** For each snapshot and size, the full AC-OPF hierarchy solves 16 cases:
   - the 5 best-ranked cases (top);
   - the 5 with p closest to 0.5 (borderline);
   - the 3 worst;
   - 3 random.

   That is 96 solver runs in total.
4. **Report.** Outputs are a confusion matrix, Δcost error, and a map (`siting_map_300mw.png`).

## Results

### Cost
| | Value |
|---|---|
| Screen, model | 4,914 cases × 0.39 s median (1,936 s total on 4 CPU threads) |
| Brute force, solver | Estimated 54 solver-hours (39.6 s median per verified case, because stressed cases need several relaxation levels) |

### Predicted feasible share
| Size | Winter peak | Summer peak |
|---|---|---|
| 100 MW | 97.7% | 96.8% |
| 300 MW | 46.0% | 40.2% |
| 500 MW | 32.5% | 12.3% |

### Solver check (96 cases)
Positive = feasible at nominal limits.

| Group | n | Solver feasible | Model says feasible | Agree |
|---|---|---|---|---|
| top-ranked | 30 | 12 | 30 | 12 |
| borderline | 30 | 8 | 14 | 16 |
| random | 18 | 8 | 7 | **17** |
| worst | 18 | 0 | 0 | **18** |

| Size | n | Solver feasible | Model says feasible | Agree | AUC |
|---|---|---|---|---|---|
| 100 MW | 32 | 21 | 21 | 24 | 0.84 |
| 300 MW | 32 | 4 | 16 | 18 | 0.69 |
| 500 MW | 32 | 3 | 14 | 21 | 0.92 |

Overall the model and solver agree on 63 of 96 cases (65.6%). The confusion matrix is tp 23, fp 28, fn 5, tn 40, and the ranking
AUC is 0.80. The sample is deliberately enriched with hard cases (top and borderline), so these figures understate random-sample
accuracy: on the random cases the model and solver agree on 17 of 18.

On the 28 solver-feasible cases, the Δcost error (MAE) is 18.5 k$/h and the correlation is only 0.26. The median true marginal
cost is 65.5 $/MWh of new load.

## Findings
1. **Ruling out bad sites works.** All 18 worst-ranked cases are confirmed infeasible, and random cases agree 17 of 18. For 100 MW,
   the model and solver give the same feasible share.
2. **The top of the ranking is over-optimistic (winner's curse).** All 30 top-ranked cases were predicted feasible, but only 12
   really are. The `large_load` family is 20% of the scenarios, so only about 200 large-load cases were in
   the 1,000 training scenarios. That is spread over 819 buses × 3 sizes × varying hours, so most site/size combinations were never
   seen, and the two peak snapshots sit at the top of the load range. The cases with the most optimistic predictions are exactly where the model errs most.
   This is the same false-positive mode seen in Phase 6 (6 false positives on the test set), made larger by extrapolation.
3. **The Δcost ranking is weak** (correlation 0.26). The model's cost error of about 4% of system cost (~10–13 k$/h) is as large
   as the effect it is trying to measure (~7–35 k$/h). The model is not good enough to rank sites by cost, only by feasibility.
4. **Use the model as a filter, not an answer.** Use it to discard clearly infeasible sites, which removes 54–88% of the 300
   and 500 MW candidates. Send everything that is not clearly ruled out to the solver. Applying the solver to the model's
   "p ≥ 0.5" set alone would still leave many infeasible picks, so every shortlisted site must be solved.
5. **Improvements:**
   - Add many more single-bus large-load scenarios (100–600 MW across 138/240 kV buses) to fine-tuning.
   - Calibrate p on that family.
   - Re-run the solver check.

   Because this is a synthetic grid, the absolute numbers mean nothing for real Alberta sites.

## Files
- `screen_predictions.csv`: all 4,914 cases, with bus name, kV and coordinates.
- `verify_results.csv`: the 96 solver checks, including relaxation level, true cost and Δcost.
- `siting_summary.json`.
- `siting_map_300mw.png`: P(feasible) for 300 MW at both snapshots. Rings show solver checks (green = ok, red = infeasible).

To reproduce:
```bash
.venv/bin/python -m abgrid.usecase.siting_screen base            # base snapshots (AC-OPF), graph exports
.venv-fm/bin/python -I src/abgrid/usecase/siting_screen.py screen  # model screen (GridSFM env)
.venv/bin/python -m abgrid.usecase.siting_screen verify          # solver check
.venv/bin/python -m abgrid.usecase.siting_screen report
```
