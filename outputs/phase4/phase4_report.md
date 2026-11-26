# Phase 4: exports, verification and relaxation hierarchy

> Approximate open-data model, NOT AESO's grid. Costs illustrative.

## Exports (one per-unit case, three writers)

- `outputs/phase4/alberta_base.m` — MATPOWER v2 (gencost model 2, $/MW² h, $/MWh)
- `outputs/phase4/alberta_base.powermodels.json` — PowerModels network dict (per-unit, radians)
- `outputs/phase4/alberta_base.pyg.json` — GridSFM / OPFData (widths 4/11/9/11) with solution block

All three are generated from the same arrays (`abgrid.export.case.build_case`) and agree exactly; the GridSFM file matches the Phase 3 converter bit-for-bit.

## MATPOWER round trip

The `.m` text is parsed back, rebuilt with pandapower `from_ppc`, and re-solved from a flat start.

| | original | round trip |
|---|---|---|
| objective | 280752.6 | 280844.3 (rel. diff 3.27e-04) |
| total generation MW | 12506.3 | 12507.5 |
| buses / branches / units | 982 / 1289 / 253 | 982 / 1289 / 253 |
| max |ΔV| pu | — | 0.0185 (flat start reaches a nearby optimum) |

## Sanity checks (base case)

- angle_diff_max_deg: 27.12
- angle_diff_p99_deg: 7.85
- branches_angle_gt_30deg: 0
- lmp_dc_ac_correlation: 0.903
- gen_minus_load_mw_ac: 327.8
- losses_%_ac: 2.99

## Relaxation hierarchy (load scaling stress test)

Levels are cumulative and applied only if the previous level is infeasible: 0 nominal → 1 line ratings ×1.2 → 2 voltage 0.90–1.10 → 3 ±200 MVAr support at ≥240 kV buses → 4 load shedding at VOLL 5000. The level reached is stored with each case as a quality signal.

|   load_factor |   load_mw | feasible   |   relaxation_level | relaxation_name     |   seconds_total |   losses_% |   load_shed_mw |   lmp_max |   vm_min |
|--------------:|----------:|:-----------|-------------------:|:--------------------|----------------:|-----------:|---------------:|----------:|---------:|
|           1   |     10978 | True       |                  0 | nominal             |             9.8 |       2.99 |            0   |    138.01 |     0.95 |
|           1.1 |     12076 | True       |                  0 | nominal             |            10.5 |       3.06 |            0   |    145.84 |     0.95 |
|           1.2 |     13174 | True       |                  1 | line_ratings_x1.2   |            20.5 |       3.38 |            0   |    149.42 |     0.95 |
|           1.3 |     14272 | True       |                  1 | line_ratings_x1.2   |            23.7 |       3.21 |            0   |    221.48 |     0.95 |
|           1.4 |     15370 | True       |                  4 | load_shedding_slack |            51   |       3.34 |          224.9 |   9027.39 |     0.9  |
|           1.5 |     16468 | False      |                nan |                     |            55.5 |     nan    |          nan   |    nan    |   nan    |

Line-loading flags at levels ≥1 are measured against the original (unrelaxed) ratings.
At ×1.5 load the available capacity at the snapshot (wind ≈2.9 GW) is below demand, so no level is feasible: the case is labelled infeasible.

## Solver

pandapower 3.5 / PYPOWER interior point (PIPS) in Python. Julia + PowerModels.jl/Ipopt is not installed; the PowerModels JSON is ready for cross-validation if Julia is added.