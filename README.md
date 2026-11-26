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
| 2. Bus-branch topology | in review — `outputs/phase2/phase2_report.md` |
| 3–7 | not started |

## Reproducing
```bash
uv venv -p 3.12 .venv && . .venv/bin/activate && uv pip install -e ".[dev]"
python -m abgrid.cli collect            # downloads into data/raw/ (cached)
python -m abgrid.cli report-phase1      # outputs/phase1/
python -m abgrid.cli build-topology     # data/interim/topology/
python -m abgrid.cli report-phase2      # outputs/phase2/ (maps, provenance summary)
pytest -q                               # topology tests
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

## Limitations
Approximate topology, synthetic electrical parameters, illustrative cost curves.
Not for operational use. Details will be added per phase.
