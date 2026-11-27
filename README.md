# Approximate open-data Alberta transmission grid for AC-OPF and grid foundation models

> **Disclaimer.** This is a personal research project. The network built here is an
> **approximation** of Alberta's high-voltage system assembled from open data plus
> synthetic engineering defaults. Real electrical parameters, ratings and topology
> details are not public. **This is not AESO's actual grid model**, it is not
> endorsed by AESO or the Government of Alberta, and it must not be used for
> operational, planning, market or investment decisions.

## Purpose
1. Build a bus-branch model of the Alberta HV grid (≥69 kV) from open data.
2. Make it solvable with DC- and AC-OPF.
3. Generate training scenarios (load/renewable variation, N-1 outages, large new loads).
4. Evaluate and fine-tune open grid foundation models (GridSFM, gridfm-graphkit).
5. Use case: screening data-centre siting.

Every bus/line/transformer/generator/load attribute carries a provenance flag:
`open:<dataset>`, `derived:<method>` or `synthetic:<rule>`. A provenance report
states what share of each attribute came from open data.

## Status
| Phase | Status |
|---|---|
| 1. Collect open data | done except OSM (extract pending upload) — `outputs/phase1/` |
| 2. Bus-branch topology | done — `outputs/phase2/phase2_report.md` |
| 3. Electrical parameters + base case | done — `outputs/phase3/phase3_report.md` (DC- and AC-OPF solve at relaxation level 0); GridSFM zero-shot smoke test run |
| 4. Exports, verification, relaxation hierarchy | done — `outputs/phase4/phase4_report.md` (PowerModels + Ipopt cross-check matches) |
| 5. Scenarios | done — `outputs/phase5/phase5_report.md`; calibrated rerun `s1250c` (1,250) after cost calibration (`data/interim/params/calibration.json`) |
| 6. GridSFM zero-shot vs fine-tuned | done — `outputs/phase6/phase6_report.md` (FT-1000 + θ refine: cost error 4.45%, θ 2.2°, infeasible F1 0.83; warm start gives no speed-up; gridfm-graphkit skipped, no public weights) |
| 7. Data-centre siting | done (in review) — `outputs/phase7/phase7_report.md` (4,914-case screen + 96 solver checks); interactive prototype in `outputs/visual/` |

## Reproducing
```bash
uv venv -p 3.12 .venv && . .venv/bin/activate && uv pip install -e ".[dev]"
python -m abgrid.cli collect            # downloads into data/raw/ (cached)
python -m abgrid.cli report-phase1      # outputs/phase1/
python -m abgrid.cli build-topology     # data/interim/topology/
python -m abgrid.cli report-phase2      # outputs/phase2/ (maps, provenance summary)
python -m abgrid.cli build-params       # Phase 3 parameters + pandapower net
python -m abgrid.cli solve-base         # DC-OPF then AC-OPF (relaxation hierarchy)
python -m abgrid.cli report-phase3
python -m abgrid.cli phase4             # MATPOWER / PowerModels / GridSFM exports, round trip, stress test
python -m abgrid.cli scenarios          # Phase 5: config `scenarios` block (n, families, seed)
julia src/abgrid/julia/solve_pm.jl outputs/phase4/alberta_base.m out.json   # PowerModels cross-check
pytest -q                               # topology tests

# Cost calibration (inverse OPF on 2024 metered dispatch) -> data/interim/params/calibration.json
python -c "from abgrid.config import load_config; from abgrid.params.calibrate import run; run(load_config())"

# Phase 6-7 run in a separate GridSFM environment (.venv-fm: torch CPU + torch_geometric + gridsfm)
.venv-fm/bin/python -I src/abgrid/fm/finetune.py --rounds 10,100,1000   # zero-shot + fine-tuning rounds
.venv-fm/bin/python -I src/abgrid/fm/refine.py                          # θ-weighted continuation (Phase 6b)
.venv-fm/bin/python -I src/abgrid/fm/warmstart.py predict 30            # warm-start test (Phase 6c) ...
.venv/bin/python -m abgrid.fm.warmstart solve                           # ... solver side
.venv/bin/python -m abgrid.usecase.siting_screen base                   # Phase 7 base snapshots
.venv-fm/bin/python -I src/abgrid/usecase/siting_screen.py screen       # Phase 7 model screen
.venv/bin/python -m abgrid.usecase.siting_screen verify                 # Phase 7 solver check
.venv/bin/python -m abgrid.usecase.siting_screen report
```
CODERS is optional and needs `CODERS_API_KEY` (sent as a `key` URL parameter over HTTPS;
scrubbed from logs and the manifest).
All settings live in `config/alberta.yaml`. Raw data is not committed;
`data/raw/MANIFEST.jsonl` lists every downloaded file with URL, query parameters,
retrieval time and SHA-256. The AESO API key is read from `AESO_API_KEY`
(header `API-KEY`); it is never logged or written to disk.

## Data sources and licences (attributions)
| Source | Use | Licence / terms |
|---|---|---|
| Government of Alberta, *Powerlines* layer (`geospatial.alberta.ca/titan/.../access_utility/MapServer/15`) | authoritative line geometry | Contains information licensed under the **Open Government Licence – Alberta** |
| OpenStreetMap power features (via Overpass API) | voltage, circuits, substations, plants | © OpenStreetMap contributors, **ODbL 1.0**. Derived databases must remain under ODbL |
| AESO public API (asset list, metered volume, Alberta Internal Load) | generators, loads, load shapes | AESO public data, subject to AESO API terms of use |
| AESO Data Requests: Planning Area hourly load; Planning Area boundaries | spatial load allocation | AESO public data, subject to AESO website terms of use |
| CODERS (SESIT, University of Victoria / Energy Modelling Hub), `api.sesit.ca` | topology backbone: circuits, node coordinates, voltage, length, reactance, TTC; generators | registered API access; subject to CODERS terms of use (cite CODERS) |
| Cheng & Chen, *Open Datasets for Grid Modeling and Visualization: An Alberta Power Network Case*, arXiv:2504.07870; data at github.com/BenCheng2/CarbonDistributionMap | cross-check of substation/line voltages and IDs, generator–substation map | **no licence file in the repository**: used for personal research only; not redistributed |

Attribution required by OSM: *"© OpenStreetMap contributors"*, with a link to
https://www.openstreetmap.org/copyright.

## Topology method (Phase 2)
- Backbone: CODERS node-to-node circuits (1,171 AC branches ≥69 kV). One bus per
  substation per voltage class (69/72→69, 138/144→138, 240, 500).
- Transformers: placed between adjacent voltage classes at the same substation
  (`derived:multi_voltage_node_adjacent_classes`); parameters in Phase 3.
- HVDC WATL (Genesee–Langdon) and EATL (Heathfield–Newell) are not in CODERS;
  added from config with endpoints/ratings flagged `synthetic:config_hvdc_*`.
- Interties BC (4 tie points), SK (McNeill), MATL: boundary buses with
  configurable, illustrative import/export limits.
- Geometry: shortest path on the stitched Government of Alberta powerline
  network between substations, accepted if consistent with the CODERS length or
  with crow-fly distance; otherwise a synthetic straight line.
- Data-quality findings are flagged, never silently fixed: CODERS segment
  lengths inconsistent with GoA routes; 7 nodes with likely coordinate errors;
  arXiv coordinates offset by tens–hundreds of km (used only for ID/voltage checks).

## Parameters and assumptions (Phase 3)
- Line X and thermal ratings (winter/summer TTC) are open CODERS data; R from X
  with textbook X/R by class; B from textbook µS/km × route length.
- Transformer impedances/ratings, generator reactive limits, p_min and **cost
  curves are synthetic and illustrative** (see `config/alberta.yaml`).
- Loads: AESO planning-area hourly `ActLoad` allocated to substations by
  connected line capacity; fixed power factor 0.95.
- HVDC links modelled as fixed schedules (sending load / receiving generator).
- Solver: pandapower 3.5 / PYPOWER interior point (requires scipy < 1.14).
- Provenance across all tracked attributes: `outputs/provenance_report.csv`.

## Results summary (Phases 5–7)
All numbers are for this approximate model. They show the method and are not facts about the real Alberta system.

- **Scenarios (Phase 5).** 1,250 AC-OPF-labelled scenarios (run `s1250c`):
  - families: base hourly, N-1 line, N-1 generator, large new load;
  - 1,091 solve at nominal limits; 138 need a relaxation level; 21 are infeasible.
- **GridSFM (Phase 6).** Test split of 149 scenarios:
  - Zero-shot: cost error 18.7%, and it labels every infeasible case feasible.
  - Fine-tuned on 1,000 scenarios plus an angle-weighted continuation:
    - cost error 4.45%;
    - errors: V 0.013 pu, θ 2.2°, Pg 3.2 MW;
    - feasibility accuracy 0.96, infeasible F1 0.83, ECE 0.02;
    - about 18–24× faster than the solver on CPU.
  - Using the prediction as the starting point for the interior-point AC-OPF gave **no** speed-up.
- **Siting screen (Phase 7).** 4,914 site × size × peak cases screened in about 32 min, against an estimated 54 h by brute force.
  - The model reliably rules out bad sites.
  - Its top picks for 300–500 MW are over-optimistic: only 12 of 30 are confirmed by the solver.
  - Its cost ranking is weak (correlation 0.26).
  - Recommended workflow: model as a pre-filter, solver for every shortlisted site.

## Limitations
- **Not AESO's grid.** The topology comes from open line and substation geometry with heuristic conflation. Many lines, ratings,
  impedances, transformer data and the reactive compensation are synthetic engineering defaults (see the provenance report). The
  real parameters are not public.
- **Costs are illustrative.** Generator cost curves are calibrated to 2024 metered dispatch by inverse OPF, but they are not offers,
  and the LMPs are not pool prices.
- **Simplifications.** Interties are modelled as boundary generators, HVDC as a fixed load+generator pair, distribution as
  aggregated loads. There are no protection, stability or short-circuit checks, and only N-1 cases were sampled.
- **ML evaluation is in-distribution.** Train and test come from the same generator and topology. Large single-site loads are
  under-represented, so model accuracy on siting questions is lower than on the test split.
- **Not for operational, planning, market, interconnection or investment decisions.**
