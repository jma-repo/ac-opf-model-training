# Phase 5: scenarios (run s1000)

> Approximate open-data model, NOT AESO's grid. Costs illustrative.

1,000 scenarios, seed 20261127, 4 processes, 3,912 s wall (3.9 s/scenario). Each = a real 2024 hour (AESO planning-area load + metered wind/solar of 95 units) + a perturbation family; HVDC and intertie schedules sampled.

## Feasibility by family

| family      |   n |   feasible_nominal |   feasible_any_level |   dc_feasible |
|:------------|----:|-------------------:|---------------------:|--------------:|
| base_hourly | 400 |              1     |                 1    |         0.962 |
| large_load  | 200 |              0.54  |                 0.91 |         0.55  |
| n1_gen      | 150 |              1     |                 1    |         0.987 |
| n1_line     | 250 |              0.996 |                 1    |         0.968 |

`feasible_nominal` = AC-OPF feasible with no relaxation (used as ground truth for fine-tuning). `feasible_any_level` = feasible after some relaxation.

## Relaxation level reached

| relaxation_level   |   count |
|:-------------------|--------:|
| infeasible         |      18 |
| 0                  |     907 |
| 1                  |       6 |
| 2                  |       7 |
| 3                  |       2 |
| 4                  |      60 |

## Data-centre loads: feasible at nominal limits by connection voltage

|   kv |   dc_mw |   mean |   size |
|-----:|--------:|-------:|-------:|
|  138 |     100 |   0.98 |     54 |
|  138 |     300 |   0.25 |     48 |
|  138 |     500 |   0.09 |     58 |
|  240 |     100 |   0.95 |     19 |
|  240 |     300 |   1    |      9 |
|  240 |     500 |   0.92 |     12 |

138 kV connection points rarely take 300 MW or more; 240 kV points mostly do.

## Solve time (s)

|       |    dc_s |   ac_s_nominal |   ac_s_total |   wall_s |
|:------|--------:|---------------:|-------------:|---------:|
| count | 1000    |        1000    |      1000    |  1000    |
| mean  |    0.83 |          10.08 |        14.42 |    15.62 |
| std   |    0.23 |           5.97 |        16.15 |    16.42 |
| min   |    0.18 |           2.08 |         4.24 |     5.14 |
| 50%   |    0.81 |           8.09 |         8.46 |     9.6  |
| 90%   |    1.01 |          17.63 |        28.98 |    30.79 |
| 99%   |    1.6  |          32.37 |        85.63 |    88.03 |
| max   |    2.69 |          52.98 |       118.03 |   120.44 |

## Feasible cases: sanity

- AC losses 1.59–4.98% (median 2.33%)
- voltages 0.95–1.05 pu; load shed in nominal-feasible cases: 0
- load 8,004–12,197 MW; illustrative cost 130,301–494,037 CAD/h

## Known issue

- 28 scenarios are AC-feasible but the DC-OPF (PIPS) did not converge, mostly high-load January hours. DC labels are therefore not a reliable infeasibility signal; to be cross-checked with PowerModels DC-OPF.

Outputs: `data/interim/scenarios/s1000/` (specs.jsonl, labels.csv, 1,000 OPFData .pyg.json; not committed, 482 MB). Labels copied to `outputs/phase5/labels_s1000.csv`.