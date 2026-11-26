"""Phase 1 report: record counts per source, coverage maps (static PNG and an
interactive folium map). Reads only cached data under data/raw."""
from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import path  # noqa: E402


def _goa(raw: Path):
    f = raw / "goa_powerlines" / "powerlines.geojson"
    if not f.exists():
        return None
    g = gpd.read_file(f)
    if g.crs is None:
        g = g.set_crs("EPSG:4269")
    return g


def _osm_counts(raw: Path):
    tiles = sorted((raw / "osm" / "tiles").glob("*.json")) if (raw / "osm" / "tiles").exists() else []
    if not tiles:
        return None
    seen, counts, volt = set(), {}, {}
    for t in tiles:
        for el in json.loads(t.read_text()).get("elements", []):
            k = (el["type"], el["id"])
            if k in seen:
                continue
            seen.add(k)
            p = el.get("tags", {}).get("power")
            counts[p] = counts.get(p, 0) + 1
            if p in ("line", "minor_line", "cable"):
                volt[p] = volt.get(p, 0) + ("voltage" in el.get("tags", {}))
    return {"tiles": len(tiles), "counts": counts, "with_voltage": volt}


def run(cfg: dict):
    raw = path(cfg, "raw_dir")
    out = path(cfg, "output_dir") / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    goa = _goa(raw)
    if goa is not None:
        km = goa.to_crs(cfg["region"]["crs_projected"]).length.sum() / 1000
        rows.append(("GoA Powerlines (MapServer/15)", "line features", len(goa), f"{km:,.0f} km total"))
        for col in ("FEATURE_TYPE", "GEO_SOURCE"):
            if col in goa:
                vc = goa[col].value_counts().head(8)
                rows.append(("GoA Powerlines", f"{col} values", int(vc.sum()),
                             "; ".join(f"{k}={v}" for k, v in vc.items())))

    osm = _osm_counts(raw)
    if osm:
        for k, v in sorted(osm["counts"].items(), key=lambda x: -x[1]):
            note = f"{osm['with_voltage'].get(k, '')} with voltage tag" if k in osm["with_voltage"] else ""
            rows.append(("OSM (Overpass)", f"power={k}", v, note))
    else:
        rows.append(("OSM (Overpass)", "-", 0, "NOT COLLECTED: egress blocked"))

    a = raw / "aeso"
    if (a / "assetlist.json").exists():
        al = pd.DataFrame(json.loads((a / "assetlist.json").read_text()))
        rows.append(("AESO asset list", "assets", len(al),
                     "; ".join(f"{k}={v}" for k, v in al.asset_type.value_counts().items())))
        act = al[al.operating_status == "Active"]
        rows.append(("AESO asset list", "active assets", len(act),
                     "; ".join(f"{k}={v}" for k, v in act.asset_type.value_counts().items())))
    if (a / "ail_hourly.csv").exists():
        ail = pd.read_csv(a / "ail_hourly.csv")
        x = pd.to_numeric(ail.alberta_internal_load, errors="coerce")
        rows.append(("AESO AIL", "hourly rows", len(ail),
                     f"{ail.begin_datetime_mpt.min()} .. {ail.begin_datetime_mpt.max()}; "
                     f"min {x.min():.0f} / mean {x.mean():.0f} / max {x.max():.0f} MW"))
    for f in sorted(a.glob("metered_volume_*.csv.gz")):
        mv = pd.read_csv(f)
        rows.append(("AESO metered volume", f.name.replace(".csv.gz", ""), len(mv),
                     f"{mv.asset_ID.nunique()} assets; "
                     + "; ".join(f"{k}={v}" for k, v in
                                 mv.groupby('asset_class').asset_ID.nunique().items())))

    af = raw / "aeso_files"
    pa = None
    for f in sorted(af.glob("Planning-Area-Load-Hourly-*/*.csv")):
        d = pd.read_csv(f, usecols=["DT_MST", "PLANNING_AREA", "ActLoad"])
        rows.append(("AESO planning-area hourly load", f.parent.name[-4:], len(d),
                     f"{d.PLANNING_AREA.nunique()} areas"))
    shp = list(af.glob("AESO-Planning-Areas-*/*.shp"))
    if shp:
        pa = gpd.read_file(shp[0])
        rows.append(("AESO planning-area boundaries", "polygons", len(pa),
                     f"{pa.REGION.nunique()} regions"))

    rep = pd.DataFrame(rows, columns=["source", "item", "count", "notes"])
    rep.to_csv(out / "record_counts.csv", index=False)
    (out / "record_counts.md").write_text(rep.to_markdown(index=False))

    # static coverage map
    fig, ax = plt.subplots(figsize=(7, 10))
    if pa is not None:
        pa.to_crs(4326).plot(ax=ax, facecolor="none", edgecolor="#bbb", lw=0.6)
    if goa is not None:
        goa.to_crs(4326).plot(ax=ax, color="#c0392b", lw=0.4, label="GoA powerlines")
    ax.set_title("Phase 1 coverage: GoA powerlines over AESO planning areas\n"
                 "(approximate open-data model, NOT AESO's grid)", fontsize=9)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.savefig(out / "coverage_map.png", dpi=150, bbox_inches="tight")

    # interactive map
    m = folium.Map(location=[54.5, -114.5], zoom_start=5, tiles="OpenStreetMap")
    if pa is not None:
        folium.GeoJson(pa.to_crs(4326), name="AESO planning areas",
                       style_function=lambda _: {"color": "#888", "weight": 1, "fillOpacity": 0},
                       tooltip=folium.GeoJsonTooltip(["NAME", "Area_ID", "REGION"])).add_to(m)
    if goa is not None:
        g2 = goa.to_crs(4326)[["OBJECTID", "geometry"]].copy()
        g2["geometry"] = g2.geometry.simplify(0.0005)
        folium.GeoJson(g2, name="GoA powerlines",
                       style_function=lambda _: {"color": "#c0392b", "weight": 1.5}).add_to(m)
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:8px;left:8px;z-index:9999;background:#fff;"
        "padding:4px 8px;font:12px sans-serif'>Approximate open-data model, NOT AESO's grid. "
        "Powerlines: Gov. of Alberta (OGL-Alberta). Planning areas: AESO.</div>"))
    m.save(out / "coverage_map.html")
    print(rep.to_string(index=False))
    return rep
