# Phase 3 parameters and base case

> Approximate open-data model, NOT AESO's grid. Cost curves are illustrative only.

Snapshot: **2024-01-25 18:00:00 MPT**, winter ratings.

## Defaults used where open data is missing

Line R and B (X and ratings come from CODERS). R = X_open / (X/R of the class).

|   kV class |   r_ohm_per_km |   x_ohm_per_km |   b_us_per_km |
|-----------:|---------------:|---------------:|--------------:|
|         69 |          0.21  |           0.45 |           3.6 |
|        138 |          0.12  |           0.48 |           3.4 |
|        240 |          0.05  |           0.49 |           3.4 |
|        500 |          0.028 |           0.33 |           5.2 |

Sources/assumptions: Kundur, *Power System Stability and Control*, Table 6.1 (230/500 kV); typical 266/477 kcmil ACSR for 69/138 kV (Glover, Sarma & Overbye, App. A).
Open CODERS X per km (median by class, ohm/km): {'69': 0.386, '138': 0.434, '240': 0.477, '500': 0.395} — consistent with the textbook range, which supports using open X directly.

Transformers (synthetic by voltage pair; rating raised to 0.5×min(HV, LV connected line ratings) where larger; 18 upsized):

| pair    |   sn_mva |   vk_percent |   vkr_percent |
|:--------|---------:|-------------:|--------------:|
| 500/240 |     1200 |           14 |          0.25 |
| 240/138 |      400 |           12 |          0.3  |
| 240/69  |      150 |           10 |          0.4  |
| 138/69  |      100 |            9 |          0.5  |
| 500/138 |      600 |           13 |          0.3  |

Generator cost and reactive limits (synthetic, illustrative):

| fuel       |   c1 |    c2 |   q_max_frac |   q_min_frac |
|:-----------|-----:|------:|-------------:|-------------:|
| wind       |    0 | 0     |         0.33 |        -0.33 |
| solar      |    0 | 0     |         0.33 |        -0.33 |
| hydro      |    8 | 0.002 |         0.5  |        -0.3  |
| biomass    |   25 | 0.005 |         0.5  |        -0.3  |
| gas_cogen  |   30 | 0.004 |         0.5  |        -0.3  |
| gas_cc     |   40 | 0.004 |         0.5  |        -0.3  |
| gas_steam  |   55 | 0.005 |         0.5  |        -0.3  |
| gas_peaker |   90 | 0.01  |         0.5  |        -0.3  |

## Generators and load

- Active units: 245 (dropped 18 retired/not yet built). Converted coal-to-gas units re-typed as gas_steam using the arXiv/AESO type.
- Installed MW by fuel: {'biomass': 444.0, 'gas_cc': 3869.0, 'gas_cogen': 5278.0, 'gas_peaker': 991.0, 'gas_steam': 3078.0, 'hydro': 927.0, 'solar': 1844.0, 'wind': 5516.0}
- Available MW at snapshot (wind/solar from AESO metered volume): {'biomass': 444.0, 'gas_cc': 3869.0, 'gas_cogen': 5278.0, 'gas_peaker': 991.0, 'gas_steam': 3078.0, 'hydro': 927.0, 'solar': 1.0, 'wind': 2943.0}
- Load: 10978.5 MW on 615 buses (AESO planning-area ActLoad; unplaced 0.0 MW), pf 0.95.
- HVDC WATL/EATL: fixed schedule {'WATL': 600, 'EATL': 600} MW north→south, 1.5% loss.
- Interties: fixed net schedule {'BC': 0, 'SK': 0, 'MATL': 0} MW.

## Base-case OPF

| | DC-OPF | AC-OPF |
|---|---|---|
| converged | True | True (relaxation level 0) |
| solve time s | 0.85 | 6.49 |
| generation MW (excl. HVDC) | 10978.5 | 11306.3 |
| AC losses MW (%) | — | 327.8 (2.99%) |
| cost (illustrative CAD/h) | 268580.2 | 280752.6 |
| max line loading (S, %) | 100.0 | 100.0 |
| voltage range pu | — | 0.95–1.05 |
| LMP min/median/max | -46.13/49.86/133.93 | -50.4/55.66/138.01 |
| flags | [] | [] |

DC-OPF median LMP by kV class: {'69.0': 50.63, '138.0': 49.85, '240.0': 51.17, '500.0': 49.63}

## Provenance (Phase 3 attributes)

| element       | attribute       |    n |   open_% |   derived_% |   synthetic_% |
|:--------------|:----------------|-----:|---------:|------------:|--------------:|
| branch_e      | x_ohm_model     | 1171 |    100   |         0   |           0   |
| branch_e      | r_ohm           | 1171 |      0   |         0   |         100   |
| branch_e      | b_us            | 1171 |      0   |         0   |         100   |
| branch_e      | length_model_km | 1171 |     85.6 |        14.4 |           0   |
| branch_e      | rate_mva        | 1171 |      0   |       100   |           0   |
| transformer_e | sn_mva          |  118 |      0   |        15.3 |          84.7 |
| transformer_e | vk_percent      |  118 |      0   |         0   |         100   |
| transformer_e | vkr_percent     |  118 |      0   |         0   |         100   |
| generator     | unit_code       |  245 |    100   |         0   |           0   |
| generator     | aeso_asset_id   |  245 |      0   |       100   |           0   |
| generator     | bus             |  245 |      0   |       100   |           0   |
| generator     | p_max_mw        |  245 |    100   |         0   |           0   |
| generator     | fuel            |  245 |      0   |       100   |           0   |
| generator     | availability    |  245 |      0   |        40   |          60   |
| generator     | p_min_mw        |  245 |      0   |         0   |         100   |
| generator     | q_max_mvar      |  245 |      0   |         0   |         100   |
| generator     | q_min_mvar      |  245 |      0   |         0   |         100   |
| generator     | cost_c1         |  245 |      0   |         0   |         100   |
| generator     | cost_c2         |  245 |      0   |         0   |         100   |
| load          | bus             |  615 |      0   |       100   |           0   |
| load          | planning_area   |  615 |      0   |       100   |           0   |
| load          | p_mw            |  615 |      0   |       100   |           0   |
| load          | q_mvar          |  615 |      0   |         0   |         100   |

Mean share over all tracked attributes (Phases 2+3): open 35.5%, derived 30.8%, synthetic 33.7%. Full table: `outputs/provenance_report.csv`.