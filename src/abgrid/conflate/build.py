"""Phase 2: build the bus-branch topology.

Backbone: CODERS node-to-node transmission circuits (open:coders).
Geometry: Government of Alberta powerlines, matched by network path
(derived:goa_path_match), else straight line (synthetic:straight_line).
Cross-checks: arXiv 2504.07870 dataset (substation and line voltages).
OSM: used for voltage cross-check when the extract is present.
Outputs go to data/interim/topology/ (tables carry provenance columns)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .. import provenance as pv
from ..config import path
from .goa_match import GoAGraph, match_branches

log = logging.getLogger("abgrid.conflate.build")

S_CODERS = "open:coders"


def _load_coders(raw: Path):
    j = lambda n: json.loads((raw / "coders" / f"{n}.json").read_text())  # noqa: E731
    nodes = pd.DataFrame(j("substations") + j("junctions"))
    lines = pd.DataFrame(j("transmission_lines"))
    gens = pd.DataFrame(j("generators"))
    ties = pd.DataFrame(j("interjurisdictional_interties"))
    for c in ("latitude", "longitude"):
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    for c in ("line_segment_length_km", "Transmission_Line_Segment_Reactance",
              "ttc_summer", "ttc_winter", "voltage", "number_of_circuits"):
        lines[c] = pd.to_numeric(lines[c], errors="coerce")
    return nodes, lines, gens, ties


def kv_class(kv, cfg):
    m = {int(k): v for k, v in cfg["topology"]["kv_class_map"].items()}
    return m.get(int(kv)) if pd.notna(kv) else None


def build(cfg: dict) -> dict:
    raw = path(cfg, "raw_dir")
    out = path(cfg, "interim_dir") / "topology"
    out.mkdir(parents=True, exist_ok=True)
    t = cfg["topology"]
    report: dict = {}

    nodes, lines, gens, ties = _load_coders(raw)
    report["coders_lines_in"] = len(lines)

    # ---------- branches (AC circuits) ----------
    lines = lines.rename(columns={"network_node_code_starting": "from_node",
                                  "network_node_code_ending": "to_node"})
    self_loops = lines[lines.from_node == lines.to_node]
    report["dropped_self_loops"] = self_loops.transmission_circuit_id.tolist()
    lines = lines[lines.from_node != lines.to_node].copy()
    below = lines[lines.voltage < t["min_kv"]]
    report["dropped_below_min_kv"] = len(below)
    lines = lines[lines.voltage >= t["min_kv"]].copy()
    lines["kv_class"] = [kv_class(v, cfg) for v in lines.voltage]
    if lines.kv_class.isna().any():
        raise ValueError(f"unmapped voltages: {lines[lines.kv_class.isna()].voltage.unique()}")

    # ---------- node coordinates ----------
    node_ll = {r.node_code: (r.longitude, r.latitude, S_CODERS)
               for r in nodes.itertuples() if pd.notna(r.latitude)}
    # intertie points have no coordinates: use the AB-side neighbour location
    for r in lines.itertuples():
        for a, b in ((r.from_node, r.to_node), (r.to_node, r.from_node)):
            if a not in node_ll and b in node_ll:
                node_ll[a] = (*node_ll[b][:2], "derived:neighbour_node_location")
    missing = set(lines.from_node) | set(lines.to_node)
    missing -= set(node_ll)
    if missing:
        raise ValueError(f"nodes without coordinates: {sorted(missing)}")

    # ---------- buses: one per (node, voltage class) ----------
    ends = pd.concat([lines[["from_node", "kv_class"]].set_axis(["node", "kv"], axis=1),
                      lines[["to_node", "kv_class"]].set_axis(["node", "kv"], axis=1)])
    bus_keys = ends.drop_duplicates()
    # HVDC endpoints attach to the highest AC class at the node
    bus_keys = bus_keys.sort_values(["node", "kv"]).reset_index(drop=True)
    bus_keys["bus"] = np.arange(1, len(bus_keys) + 1)
    key2bus = {(r.node, r.kv): r.bus for r in bus_keys.itertuples()}
    nmeta = nodes.set_index("node_code")
    tie_nodes = {n: k for k, b in t["boundary"].items() for n in b["nodes"]}

    bus = pd.DataFrame({"bus": bus_keys.bus})
    pv.put(bus, "node_code", bus_keys.node.values, S_CODERS)
    pv.put(bus, "name", [nmeta.node_name.get(n, n) if n in nmeta.index else n
                         for n in bus_keys.node], [S_CODERS if n in nmeta.index
                                                   else "derived:node_code" for n in bus_keys.node])
    pv.put(bus, "lon", [node_ll[n][0] for n in bus_keys.node], [node_ll[n][2] for n in bus_keys.node])
    pv.put(bus, "lat", [node_ll[n][1] for n in bus_keys.node], [node_ll[n][2] for n in bus_keys.node])
    pv.put(bus, "base_kv", bus_keys.kv.values, "derived:kv_class_of_open_line_voltage")
    pv.put(bus, "node_type", [nmeta.node_type.get(n) if n in nmeta.index else "Intertie"
                              for n in bus_keys.node],
           [S_CODERS if n in nmeta.index else "derived:intertie_table" for n in bus_keys.node])
    pv.put(bus, "is_boundary", [n in tie_nodes for n in bus_keys.node], "open:coders_interties")

    # planning area via spatial join with AESO planning-area polygons
    pa_shp = next((raw / "aeso_files").glob("AESO-Planning-Areas-*/*.shp"))
    pa = gpd.read_file(pa_shp).to_crs(4326)
    pts = gpd.GeoDataFrame(bus[["bus"]], geometry=gpd.points_from_xy(bus.lon, bus.lat), crs=4326)
    j = gpd.sjoin(pts, pa[["Area_ID", "NAME", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    # points just outside a polygon (border nodes): nearest polygon
    miss = j.Area_ID.isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss].to_crs(t_crs := cfg["region"]["crs_projected"]),
                                 pa[["Area_ID", "geometry"]].to_crs(t_crs), how="left")
        near = near[~near.index.duplicated()]
        j.loc[miss, "Area_ID"] = near.Area_ID
    pv.put(bus, "planning_area", j.Area_ID.astype(str).values,
           np.where(miss, "derived:nearest_aeso_planning_area", "derived:within_aeso_planning_area"))

    br = pd.DataFrame({"branch": np.arange(1, len(lines) + 1)})
    lines = lines.reset_index(drop=True)
    pv.put(br, "from_bus", [key2bus[(a, k)] for a, k in zip(lines.from_node, lines.kv_class)],
           "derived:coders_node_and_kv_class")
    pv.put(br, "to_bus", [key2bus[(b, k)] for b, k in zip(lines.to_node, lines.kv_class)],
           "derived:coders_node_and_kv_class")
    pv.put(br, "circuit_id", lines.transmission_circuit_id.values, S_CODERS)
    pv.put(br, "voltage_kv", lines.voltage.values, S_CODERS)
    pv.put(br, "length_km", lines.line_segment_length_km.values, S_CODERS)
    pv.put(br, "n_circuits", lines.number_of_circuits.values, S_CODERS)
    pv.put(br, "x_ohm", lines.Transmission_Line_Segment_Reactance.values,
           np.where(lines.Transmission_Line_Segment_Reactance.notna(), S_CODERS, "synthetic:missing"))
    pv.put(br, "rate_summer_mw", lines.ttc_summer.values, S_CODERS)
    pv.put(br, "rate_winter_mw", lines.ttc_winter.values, S_CODERS)
    br["from_node"], br["to_node"], br["kv_class"] = lines.from_node, lines.to_node, lines.kv_class
    br["owner"] = lines.owner

    # ---------- transformers between voltage classes at the same node ----------
    tr_rows = []
    for node, grp in bus_keys.groupby("node"):
        kvs = sorted(grp.kv, reverse=True)
        for hv, lv in zip(kvs[:-1], kvs[1:]):        # chain adjacent classes
            tr_rows.append((key2bus[(node, hv)], key2bus[(node, lv)], hv, lv, node))
    tr = pd.DataFrame(tr_rows, columns=["hv_bus", "lv_bus", "hv_kv", "lv_kv", "node_code"])
    tr.insert(0, "transformer", np.arange(1, len(tr) + 1))
    for c in ("hv_bus", "lv_bus", "hv_kv", "lv_kv"):
        tr[pv.src(c)] = "derived:multi_voltage_node_adjacent_classes"

    # ---------- HVDC ----------
    dc_rows = []
    for h in t["hvdc"]:
        fb = max((k for k in key2bus if k[0] == h["from_node"]), key=lambda k: k[1])
        tb = max((k for k in key2bus if k[0] == h["to_node"]), key=lambda k: k[1])
        dc_rows.append({"dc_line": len(dc_rows) + 1, "name": h["name"],
                        "from_bus": key2bus[fb], "to_bus": key2bus[tb], "p_max_mw": h["p_max_mw"]})
    dc = pd.DataFrame(dc_rows)
    for c in ("from_bus", "to_bus", "p_max_mw", "name"):
        dc[pv.src(c)] = "synthetic:config_hvdc_public_project_description"

    # ---------- boundary (interties) ----------
    bd_rows = []
    for k, b in t["boundary"].items():
        share = len(b["nodes"])
        for n in b["nodes"]:
            kk = [x for x in key2bus if x[0] == n]
            if not kk:
                log.warning("boundary node %s not in network", n)
                continue
            bd_rows.append({"bus": key2bus[kk[0]], "neighbour": k,
                            "import_max_mw": b["import_max_mw"] / share,
                            "export_max_mw": b["export_max_mw"] / share})
    bd = pd.DataFrame(bd_rows)
    bd["bus__src"] = "open:coders_interties"
    bd["neighbour__src"] = "open:coders_interties"
    bd["import_max_mw__src"] = "synthetic:config_intertie_limit_split_evenly"
    bd["export_max_mw__src"] = "synthetic:config_intertie_limit_split_evenly"

    # ---------- largest connected AC component ----------
    G = nx.Graph()
    G.add_nodes_from(bus.bus)
    G.add_edges_from(zip(br.from_bus, br.to_bus))
    G.add_edges_from(zip(tr.hv_bus, tr.lv_bus))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    keep = comps[0]
    report["components"] = [len(c) for c in comps]
    dropped_bus = bus[~bus.bus.isin(keep)]
    report["dropped_buses"] = dropped_bus[["bus", "node_code", "base_kv"]].to_dict("records")
    report["dropped_branches"] = br[~(br.from_bus.isin(keep) & br.to_bus.isin(keep))].circuit_id.tolist()
    bus = bus[bus.bus.isin(keep)]
    br = br[br.from_bus.isin(keep) & br.to_bus.isin(keep)]
    tr = tr[tr.hv_bus.isin(keep) & tr.lv_bus.isin(keep)]
    dc = dc[dc.from_bus.isin(keep) & dc.to_bus.isin(keep)]
    bd = bd[bd.bus.isin(keep)]

    # ---------- geometry from GoA ----------
    crs = cfg["region"]["crs_projected"]
    goa = gpd.read_file(raw / "goa_powerlines" / "powerlines.geojson")
    goa = (goa.set_crs(4269) if goa.crs is None else goa).to_crs(crs)
    gg = GoAGraph(goa, t["stitch_tol_m"])
    ll = gpd.GeoSeries([Point(v[0], v[1]) for v in node_ll.values()], crs=4326).to_crs(crs)
    node_xy = {k: (p.x, p.y) for k, p in zip(node_ll, ll)}
    m, snap_d = match_branches(gg, br.reset_index(drop=True), node_xy, cfg)
    br = br.reset_index(drop=True)
    geoms, gsrc, ratio, used, pkm, why = [], [], [], set(), [], []
    for i, r in br.iterrows():
        mm = m.get(i, {})
        ratio.append(mm.get("ratio", np.nan))
        pkm.append(mm.get("path_km", np.nan))
        why.append(mm.get("reason", "matched" if "geometry" in mm else "no_path"))
        if "geometry" in mm:
            geoms.append(mm["geometry"]); gsrc.append(mm["how"])
            used.update(mm["goa_lines"])
        else:
            geoms.append(LineString([node_xy[r.from_node], node_xy[r.to_node]]))
            gsrc.append("synthetic:straight_line")
    br = gpd.GeoDataFrame(br, geometry=gpd.GeoSeries(geoms, crs=crs)).to_crs(4326)
    br["geometry__src"] = gsrc
    br["goa_ratio"] = ratio
    br["goa_path_km"] = pkm
    br["goa_match"] = why
    # CODERS segment length inconsistent with the GoA route -> flag (not changed)
    flo, fhi = t["length_flag_ratio"]
    br["length_flag"] = br.goa_path_km.notna() & ~br.goa_ratio.between(flo, fhi)
    report["length_flagged_branches"] = int(br.length_flag.sum())
    report["suspect_nodes"] = _suspect_nodes(br, node_xy)
    goa["km"] = goa.length / 1000
    goa["used_in_model"] = goa.OBJECTID.isin(used)
    report["goa"] = {
        "vertices_stitched": gg.n_stitched,
        "branches_matched": int(br.geometry__src.str.startswith("derived:goa_path_match").sum()),
        "branches_total": len(br),
        "goa_km_total": round(goa.km.sum(), 1),
        "goa_km_used": round(goa.loc[goa.used_in_model, "km"].sum(), 1),
        "goa_lines_unused": int((~goa.used_in_model).sum()),
        "node_snap_m_median": round(float(np.median(list(snap_d.values()))), 1),
    }
    goa.loc[~goa.used_in_model].to_crs(4326).to_file(out / "goa_unmatched_lines.geojson")

    # ---------- cross-checks ----------
    report["crosscheck_arxiv"] = _crosscheck_arxiv(raw, bus, br, cfg)
    report["osm"] = _osm_status(raw)

    # ---------- validate + save ----------
    for name, df in (("bus", bus), ("branch", br), ("transformer", tr),
                     ("dc_line", dc), ("boundary", bd)):
        pv.validate(df, name)
    bus.to_csv(out / "buses.csv", index=False)
    br.drop(columns="geometry").to_csv(out / "branches.csv", index=False)
    br.to_file(out / "branches.geojson")
    tr.to_csv(out / "transformers.csv", index=False)
    dc.to_csv(out / "dc_lines.csv", index=False)
    bd.to_csv(out / "boundary.csv", index=False)
    report["counts"] = {"buses": len(bus), "branches": len(br), "transformers": len(tr),
                        "dc_lines": len(dc), "boundary_buses": len(bd),
                        "buses_by_kv": bus.base_kv.value_counts().sort_index().to_dict(),
                        "branches_by_kv": br.kv_class.value_counts().sort_index().to_dict()}
    (out / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("topology: %s", report["counts"])
    return {"bus": bus, "branch": br, "transformer": tr, "dc_line": dc, "boundary": bd,
            "report": report}


def _suspect_nodes(br, node_xy, factor=3.0, min_km=10.0):
    """Nodes where most incident lines are far shorter than the crow-fly
    distance to their other end: likely coordinate errors in the source."""
    rows = {}
    for r in br.itertuples():
        crow = np.hypot(*np.subtract(node_xy[r.from_node], node_xy[r.to_node])) / 1000
        bad = crow > max(factor * r.length_km, min_km)
        for n in (r.from_node, r.to_node):
            a = rows.setdefault(n, [0, 0]); a[0] += bad; a[1] += 1
    return sorted(n for n, (b, k) in rows.items() if b and b / k > 0.5)


def _classes(s: str, cfg) -> set:
    return {kv_class(int(x), cfg) for x in re.findall(r"\d+", str(s)) if int(x) >= 60} - {None}


def _crosscheck_arxiv(raw, bus, br, cfg):
    d = raw / "arxiv_alberta"
    if not d.exists():
        return "arXiv dataset not present"
    sub = pd.read_csv(d / "Substation.csv")
    lv = pd.read_csv(d / "LineVoltage.csv")
    node_kv = bus.groupby(bus.node_code.str.split("_").str[1]).base_kv.apply(set)
    agree = total = 0
    for r in sub.itertuples():
        if r.substation in node_kv.index:
            total += 1
            agree += bool(_classes(r.voltage, cfg) & node_kv[r.substation])
    l_agree = l_total = 0
    br_kv = br.groupby("circuit_id").kv_class.apply(set)
    for r in lv.itertuples():
        c = _classes(r.Voltage, cfg)
        if r.Line in br_kv.index and c:
            l_total += 1
            l_agree += bool(c & br_kv[r.Line])
    return {"substations_matched_by_id": total, "substations_in_arxiv": len(sub),
            "substation_voltage_agree_%": round(100 * agree / max(total, 1), 1),
            "lines_matched_by_circuit_id": l_total, "lines_in_arxiv": len(lv),
            "line_voltage_agree_%": round(100 * l_agree / max(l_total, 1), 1)}


def _osm_status(raw):
    pbf = list((raw / "osm").glob("*.osm.pbf")) if (raw / "osm").exists() else []
    tiles = list((raw / "osm" / "tiles").glob("*.json")) if (raw / "osm" / "tiles").exists() else []
    if not pbf and not tiles:
        return "OSM not available yet: GoA<->OSM voltage conflation pending"
    return f"OSM present ({len(pbf)} pbf, {len(tiles)} tiles): conflation not yet run"
