"""Phase 2 report: topology stats, provenance summary, folium maps coloured by
voltage class and by provenance (open/derived vs synthetic)."""
from __future__ import annotations

import json

import folium
import geopandas as gpd
import matplotlib
import pandas as pd

from .. import provenance as pv
from ..config import path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KV_COLOURS = {69: "#2a9d8f", 138: "#457b9d", 240: "#e9a23b", 500: "#c1121f"}
PROV_COLOURS = {"derived:goa_path_match": "#1b7837", "derived:goa_path_match_crowfly": "#7fbf7b",
                "synthetic:straight_line": "#9e3d9e"}
DISCLAIMER = ("Approximate open-data model, NOT AESO's grid. Lines/substations: CODERS; "
              "geometry: Gov. of Alberta (OGL-Alberta); planning areas: AESO.")


def _load(cfg):
    d = path(cfg, "interim_dir") / "topology"
    return {"bus": pd.read_csv(d / "buses.csv"),
            "branch": gpd.read_file(d / "branches.geojson"),
            "transformer": pd.read_csv(d / "transformers.csv"),
            "dc_line": pd.read_csv(d / "dc_lines.csv"),
            "boundary": pd.read_csv(d / "boundary.csv")}, json.loads((d / "build_report.json").read_text())


def _map(t, colour_of, legend, fname, out, tooltip_cols):
    m = folium.Map(location=[54.0, -114.5], zoom_start=6, tiles="OpenStreetMap")
    br = t["branch"].copy()
    br["colour"] = br.apply(colour_of, axis=1)
    folium.GeoJson(br[["colour", *tooltip_cols, "geometry"]], name="branches",
                   style_function=lambda f: {"color": f["properties"]["colour"], "weight": 2},
                   tooltip=folium.GeoJsonTooltip(tooltip_cols)).add_to(m)
    bus = t["bus"]
    fg = folium.FeatureGroup(name="buses")
    for r in bus.drop_duplicates("node_code").itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color="#222", fill=True,
                            tooltip=f"{r.name} ({r.node_code})").add_to(fg)
    fg.add_to(m)
    bi = bus.set_index("bus")
    for r in t["dc_line"].itertuples():
        a, b = bi.loc[r.from_bus], bi.loc[r.to_bus]
        folium.PolyLine([[a.lat, a.lon], [b.lat, b.lon]], color="#000", weight=4, dash_array="8",
                        tooltip=f"HVDC {r.name} {r.p_max_mw} MW (synthetic endpoints/rating)").add_to(m)
    for r in t["boundary"].itertuples():
        a = bi.loc[r.bus]
        folium.Marker([a.lat, a.lon], tooltip=f"Intertie {r.neighbour}: import {r.import_max_mw:.0f} / "
                      f"export {r.export_max_mw:.0f} MW (synthetic limits)").add_to(m)
    items = "".join(f"<div><span style='display:inline-block;width:18px;height:4px;background:{c};"
                    f"margin-right:6px;vertical-align:middle'></span>{k}</div>" for k, c in legend.items())
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;bottom:12px;left:12px;z-index:9999;background:#fff;padding:6px 10px;"
        f"font:12px sans-serif;border:1px solid #ccc'>{items}<div style='margin-top:4px;color:#555'>"
        f"{DISCLAIMER}</div></div>"))
    folium.LayerControl().add_to(m)
    m.save(out / fname)


def run(cfg):
    out = path(cfg, "output_dir") / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    t, rep = _load(cfg)
    prov = pv.summary(t)
    prov.to_csv(out / "provenance_summary.csv", index=False)
    (out / "provenance_summary.md").write_text(prov.drop(columns="sources").to_markdown(index=False))

    tips = ["circuit_id", "voltage_kv", "length_km", "goa_match"]
    _map(t, lambda r: KV_COLOURS.get(int(r.kv_class), "#888"),
         {f"{k} kV class": c for k, c in KV_COLOURS.items()} | {"HVDC (dashed)": "#000"},
         "topology_by_voltage.html", out, tips)
    _map(t, lambda r: PROV_COLOURS.get(r.geometry__src, "#888"),
         {"geometry: GoA path (length consistent)": PROV_COLOURS["derived:goa_path_match"],
          "geometry: GoA path (CODERS length inconsistent)": PROV_COLOURS["derived:goa_path_match_crowfly"],
          "geometry: synthetic straight line": PROV_COLOURS["synthetic:straight_line"]},
         "topology_by_provenance.html", out, tips)

    fig, axes = plt.subplots(1, 2, figsize=(13, 10))
    br = t["branch"]
    for ax, (title, col, cmap) in zip(axes, [
            ("by voltage class", "kv_class", KV_COLOURS),
            ("by geometry provenance", "geometry__src", PROV_COLOURS)]):
        for k, c in cmap.items():
            s = br[br[col] == k]
            if len(s):
                s.plot(ax=ax, color=c, lw=0.6 if col == "geometry__src" else 0.4 + 0.6 * (k / 500),
                       label=str(k))
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Topology {title}", fontsize=10)
        ax.set_aspect(1 / 0.6)
    fig.suptitle("Approximate open-data Alberta grid (NOT AESO's model)", fontsize=10)
    fig.savefig(out / "topology_maps.png", dpi=150, bbox_inches="tight")

    c = rep["counts"]
    g = rep["goa"]
    lines = [
        "# Phase 2 topology report", "", f"> {DISCLAIMER}", "",
        "## Network", "",
        f"- Buses: **{c['buses']}** (by class kV: {c['buses_by_kv']})",
        f"- AC branches: **{c['branches']}** (by class kV: {c['branches_by_kv']})",
        f"- Transformers (synthetic placement between classes at one substation): **{c['transformers']}**",
        f"- HVDC links: **{c['dc_lines']}** (WATL, EATL; endpoints and ratings from config)",
        f"- Boundary (intertie) buses: **{c['boundary_buses']}**",
        f"- Connected components after build: {rep['components']}",
        f"- Dropped: self-loop circuits {rep['dropped_self_loops']}; buses outside the largest "
        f"component: {len(rep['dropped_buses'])}; branches: {len(rep['dropped_branches'])}",
        "", "## Geometry match to Government of Alberta powerlines", "",
        f"- Branches with GoA route geometry: **{g['branches_matched']} / {g['branches_total']}**",
        f"- GoA km used by matched routes: {g['goa_km_used']:,} of {g['goa_km_total']:,} km; "
        f"{g['goa_lines_unused']} GoA polylines unused (`data/interim/topology/goa_unmatched_lines.geojson`)",
        f"- Branches whose CODERS segment length disagrees with the GoA route (flagged, not changed): "
        f"{rep['length_flagged_branches']}",
        f"- Nodes with likely coordinate errors in CODERS (flagged): {rep['suspect_nodes']}",
        "", "## Cross-check against arXiv 2504.07870 dataset", "",
        *[f"- {k}: {v}" for k, v in rep["crosscheck_arxiv"].items()],
        "- arXiv coordinates are offset by tens to hundreds of km (georeferenced map image), "
        "so they are used only for ID/voltage cross-checks.",
        "", "## OSM", "", f"- {rep['osm']}",
        "", "## Provenance summary", "", prov.drop(columns="sources").to_markdown(index=False),
    ]
    (out / "phase2_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
