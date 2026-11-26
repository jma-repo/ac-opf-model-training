# Phase 2 topology report

> Approximate open-data model, NOT AESO's grid. Lines/substations: CODERS; geometry: Gov. of Alberta (OGL-Alberta); planning areas: AESO.

## Network

- Buses: **982** (by class kV: {'69': 144, '138': 657, '240': 167, '500': 14})
- AC branches: **1171** (by class kV: {'69': 156, '138': 743, '240': 255, '500': 17})
- Transformers (synthetic placement between classes at one substation): **118**
- HVDC links: **2** (WATL, EATL; endpoints and ratings from config)
- Boundary (intertie) buses: **6**
- Connected components after build: [982]
- Dropped: self-loop circuits ['215AL']; buses outside the largest component: 0; branches: 0

## Geometry match to Government of Alberta powerlines

- Branches with GoA route geometry: **677 / 1171**
- GoA km used by matched routes: 13,494.6 of 19,034.7 km; 904 GoA polylines unused (`data/interim/topology/goa_unmatched_lines.geojson`)
- Branches whose CODERS segment length disagrees with the GoA route (flagged, not changed): 169
- Nodes with likely coordinate errors in CODERS (flagged): ['AB_151S_GSS', 'AB_492S_DSS', 'AB_754S_DSS', 'AB_758S_DSS', 'AB_795S_DSS', 'AB_878S_DSS', 'AB_953S_DSS']

## Cross-check against arXiv 2504.07870 dataset

- substations_matched_by_id: 563
- substations_in_arxiv: 683
- substation_voltage_agree_%: 100.0
- lines_matched_by_circuit_id: 713
- lines_in_arxiv: 744
- line_voltage_agree_%: 99.6
- arXiv coordinates are offset by tens to hundreds of km (georeferenced map image), so they are used only for ID/voltage cross-checks.

## OSM

- OSM not available yet: GoA<->OSM voltage conflation pending

## Provenance summary

| element     | attribute      |    n |   open_% |   derived_% |   synthetic_% |
|:------------|:---------------|-----:|---------:|------------:|--------------:|
| bus         | name           |  982 |     99.4 |         0.6 |           0   |
| bus         | node_code      |  982 |    100   |         0   |           0   |
| bus         | lat            |  982 |     99.4 |         0.6 |           0   |
| bus         | lon            |  982 |     99.4 |         0.6 |           0   |
| bus         | base_kv        |  982 |      0   |       100   |           0   |
| bus         | node_type      |  982 |     99.4 |         0.6 |           0   |
| bus         | planning_area  |  982 |      0   |       100   |           0   |
| bus         | is_boundary    |  982 |    100   |         0   |           0   |
| branch      | from_bus       | 1171 |      0   |       100   |           0   |
| branch      | to_bus         | 1171 |      0   |       100   |           0   |
| branch      | circuit_id     | 1171 |    100   |         0   |           0   |
| branch      | voltage_kv     | 1171 |    100   |         0   |           0   |
| branch      | length_km      | 1171 |    100   |         0   |           0   |
| branch      | n_circuits     | 1171 |    100   |         0   |           0   |
| branch      | x_ohm          | 1171 |    100   |         0   |           0   |
| branch      | rate_summer_mw | 1171 |    100   |         0   |           0   |
| branch      | rate_winter_mw | 1171 |    100   |         0   |           0   |
| branch      | geometry       | 1171 |      0   |        57.8 |          42.2 |
| transformer | hv_bus         |  118 |      0   |       100   |           0   |
| transformer | lv_bus         |  118 |      0   |       100   |           0   |
| transformer | hv_kv          |  118 |      0   |       100   |           0   |
| transformer | lv_kv          |  118 |      0   |       100   |           0   |
| dc_line     | from_bus       |    2 |      0   |         0   |         100   |
| dc_line     | to_bus         |    2 |      0   |         0   |         100   |
| dc_line     | p_max_mw       |    2 |      0   |         0   |         100   |
| dc_line     | name           |    2 |      0   |         0   |         100   |
| boundary    | bus            |    6 |    100   |         0   |           0   |
| boundary    | import_max_mw  |    6 |      0   |         0   |         100   |
| boundary    | export_max_mw  |    6 |      0   |         0   |         100   |
| boundary    | neighbour      |    6 |    100   |         0   |           0   |