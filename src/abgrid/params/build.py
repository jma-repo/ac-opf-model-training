"""Phase 3: electrical parameters, generators and loads for one snapshot hour.

Every attribute carries provenance (see abgrid.provenance). Output:
data/interim/params/*.csv (with __src columns) and a pandapower net
(data/interim/params/net_<snapshot>.json)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandapower as pp

from .. import provenance as pv
from ..conflate.build import kv_class
from ..config import path

log = logging.getLogger("abgrid.params")


def _snapshot(cfg):
    ts = pd.Timestamp(cfg["params"]["snapshot_mpt"])
    season = "winter" if ts.month in (11, 12, 1, 2, 3, 4) else "summer"
    return ts, season


def _xr(cfg, kv):
    d = cfg["params"]["line_defaults"][int(kv)]
    return d["x_ohm_per_km"] / d["r_ohm_per_km"]


# ---------------------------------------------------------------- lines
def line_params(cfg, br: pd.DataFrame, season: str) -> pd.DataFrame:
    p = cfg["params"]
    br = br.copy()
    x = br.x_ohm.clip(lower=p["min_x_ohm"])
    pv.put(br, "x_ohm_model", x, np.where(br.x_ohm < p["min_x_ohm"], "synthetic:min_x_floor",
                                          br["x_ohm__src"]))
    pv.put(br, "r_ohm", [xi / _xr(cfg, k) for xi, k in zip(x, br.kv_class)],
           "synthetic:r_from_open_x_with_textbook_xr_ratio")
    use_goa = br.length_flag.astype(bool) & br.goa_path_km.notna()
    length = np.where(use_goa, br.goa_path_km, br.length_km)
    pv.put(br, "length_model_km", length,
           np.where(use_goa, "derived:goa_path_length_coders_length_flagged", br["length_km__src"]))
    pv.put(br, "b_us", [p["line_defaults"][int(k)]["b_us_per_km"] * L for k, L in zip(br.kv_class, length)],
           "synthetic:textbook_b_per_km_x_length")
    col = "rate_winter_mw" if season == "winter" else "rate_summer_mw"
    pv.put(br, "rate_mva", br[col].values, f"derived:coders_ttc_{season}_mw_as_mva")
    return br


# ---------------------------------------------------------------- transformers
def transformer_params(cfg, tr: pd.DataFrame, br: pd.DataFrame) -> pd.DataFrame:
    p = cfg["params"]
    tr = tr.copy()
    rate_at = pd.concat([br[["from_bus", "rate_mva"]].set_axis(["bus", "r"], axis=1),
                         br[["to_bus", "rate_mva"]].set_axis(["bus", "r"], axis=1)]).groupby("bus").r.sum()
    sn, vk, vkr, sn_src = [], [], [], []
    for r in tr.itertuples():
        key = f"{int(r.hv_kv)}/{int(r.lv_kv)}"
        d = p["transformer_defaults"][key]
        need = p["transformer_sizing_factor"] * min(rate_at.get(r.hv_bus, 0), rate_at.get(r.lv_bus, 0))
        sn.append(max(d["sn_mva"], need))
        sn_src.append("synthetic:default_by_voltage_pair" if d["sn_mva"] >= need
                      else "derived:sized_to_connected_line_ratings")
        vk.append(d["vk_percent"]); vkr.append(d["vkr_percent"])
    pv.put(tr, "sn_mva", sn, sn_src)
    pv.put(tr, "vk_percent", vk, "synthetic:default_by_voltage_pair")
    pv.put(tr, "vkr_percent", vkr, "synthetic:default_by_voltage_pair")
    return tr


# ---------------------------------------------------------------- generators
def _aeso_id(code: str) -> str:
    body = code.removeprefix("AB_").removesuffix("_GEN")
    return body[:-2] if body.endswith("00") else body


def generators(cfg, raw: Path, bus: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    p = cfg["params"]
    G = pd.DataFrame(json.loads((raw / "coders" / "generators.json").read_text()))
    yr = ts.year
    for c in ("start_year", "closure_year", "unit_installed_capacity", "latitude",
              "longitude", "network_node_voltage"):
        G[c] = pd.to_numeric(G[c], errors="coerce")
    active = (G.start_year.isna() | (G.start_year <= yr)) & (G.closure_year.isna() | (G.closure_year > yr))
    dropped = G[~active]
    G = G[active].reset_index(drop=True)
    log.info("generators active in %d: %d (dropped %d retired/not yet built)", yr, len(G), len(dropped))

    # bus mapping
    key2bus = {(r.node_code, r.base_kv): r.bus for r in bus.itertuples()}
    node_buses = bus.groupby("node_code").bus.apply(list)
    bxy = gpd.GeoSeries(gpd.points_from_xy(bus.lon, bus.lat), crs=4326).to_crs(cfg["region"]["crs_projected"])
    gb, gsrc = [], []
    for r in G.itertuples():
        k = kv_class(r.network_node_voltage, cfg)
        if (r.network_node_code, k) in key2bus:
            gb.append(key2bus[(r.network_node_code, k)]); gsrc.append("derived:coders_node_and_kv_class")
        elif r.network_node_code in node_buses.index:
            gb.append(max(node_buses[r.network_node_code])); gsrc.append("derived:coders_node_other_kv")
        else:
            pt = gpd.GeoSeries(gpd.points_from_xy([r.longitude], [r.latitude]), crs=4326) \
                .to_crs(cfg["region"]["crs_projected"]).iloc[0]
            gb.append(int(bus.bus.iloc[int(np.argmin(bxy.distance(pt).values))]))
            gsrc.append("derived:nearest_bus_by_location")

    out = pd.DataFrame({"gen": np.arange(1, len(G) + 1)})
    pv.put(out, "unit_code", G.generation_unit_code, "open:coders")
    out["aeso_asset_id"] = G.generation_unit_code.map(_aeso_id)
    out["aeso_asset_id__src"] = "derived:coders_unit_code_pattern"
    pv.put(out, "bus", gb, gsrc)
    pv.put(out, "p_max_mw", G.unit_installed_capacity, "open:coders")

    # fuel: CODERS type -> model fuel, overridden for converted coal units
    ax = pd.read_csv(raw / "arxiv_alberta" / "GeneratorSubstation.csv").set_index("Generator Name").Type
    fuel = G.gen_type.map(p["coders_fuel_map"])
    fsrc = pd.Series("derived:coders_gen_type_to_fuel", index=G.index)
    ov = out.aeso_asset_id.map(ax).map(p["arxiv_fuel_override"])
    fuel = fuel.where(ov.isna(), ov)
    fsrc = fsrc.where(ov.isna(), "derived:arxiv_aeso_type_override")
    if fuel.isna().any():
        raise ValueError(f"unmapped gen types {G.gen_type[fuel.isna()].unique()}")
    pv.put(out, "fuel", fuel.values, fsrc.values)

    # snapshot availability for wind/solar from AESO metered volume
    utc = (ts.tz_localize("America/Edmonton").tz_convert("UTC")).strftime("%Y-%m-%d %H:%M")
    mv = None
    for f in sorted((raw / "aeso").glob("metered_volume_*.csv.gz")):
        d = pd.read_csv(f)
        if (d.begin_date_utc == utc).any():
            mv = d[d.begin_date_utc == utc].groupby("asset_ID").metered_volume_mwh.sum()
            break
    avail, asrc = np.ones(len(out)), np.array(["synthetic:full_availability"] * len(out), dtype=object)
    if mv is not None:
        unit_cf = (out.aeso_asset_id.map(mv) / out.p_max_mw).clip(0, 1)
        for f in ("wind", "solar"):
            m = out.fuel == f
            known = m & unit_cf.notna()
            fleet = (out.aeso_asset_id[known].map(mv).sum() / out.p_max_mw[known].sum()) if known.any() else np.nan
            avail[known] = unit_cf[known]
            asrc[known] = "derived:aeso_metered_volume_unit"
            avail[m & ~known] = fleet if np.isfinite(fleet) else 0.3
            asrc[m & ~known] = "derived:aeso_metered_volume_fleet_cf" if np.isfinite(fleet) else "synthetic:cf_0.3"
        log.info("metered match: %d of %d units", int(unit_cf.notna().sum()), len(out))
    else:
        log.warning("no metered volume for %s; wind/solar availability synthetic", utc)
        for f, cf in (("wind", 0.35), ("solar", 0.0)):
            avail[out.fuel == f] = cf
            asrc[out.fuel == f] = f"synthetic:cf_{cf}"
    pv.put(out, "availability", avail, asrc)
    pv.put(out, "p_min_mw", out.p_max_mw * p["p_min_frac"], "synthetic:p_min_frac")
    fu = p["fuels"]
    pv.put(out, "q_max_mvar", [fu[f]["q_max_frac"] * pm for f, pm in zip(out.fuel, out.p_max_mw)],
           "synthetic:q_limits_frac_of_p_max_by_fuel")
    pv.put(out, "q_min_mvar", [fu[f]["q_min_frac"] * pm for f, pm in zip(out.fuel, out.p_max_mw)],
           "synthetic:q_limits_frac_of_p_max_by_fuel")
    pv.put(out, "cost_c1", [fu[f]["c1"] for f in out.fuel], "synthetic:illustrative_cost_by_fuel")
    pv.put(out, "cost_c2", [fu[f]["c2"] for f in out.fuel], "synthetic:illustrative_cost_by_fuel")
    out.attrs["dropped"] = len(dropped)
    return out


# ---------------------------------------------------------------- loads
def loads(cfg, raw: Path, bus: pd.DataFrame, br: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    p = cfg["params"]
    f = next((raw / "aeso_files").glob(f"Planning-Area-Load-Hourly-{ts.year}/*.csv"))
    pa = pd.read_csv(f)
    pa = pa[pd.to_datetime(pa.DT_MPT) == ts]
    area_mw = pa.groupby("PLANNING_AREA")[p["load_column"]].sum()
    area_mw.index = area_mw.index.astype(int)
    bus = bus.copy()
    bus["area"] = pd.to_numeric(bus.planning_area, errors="coerce").astype("Int64")
    size = pd.concat([br[["from_bus", "rate_mva"]].set_axis(["bus", "r"], axis=1),
                      br[["to_bus", "rate_mva"]].set_axis(["bus", "r"], axis=1)]).groupby("bus").r.sum()
    node_size = bus.assign(s=bus.bus.map(size).fillna(0)).groupby("node_code").s.sum()
    # load points: lowest-voltage bus at distribution / industrial substations
    cand = bus[bus.node_type.isin(["Distribution", "Industrial"]) & ~bus.is_boundary]
    cand = cand.sort_values("base_kv").drop_duplicates("node_code")
    rows = []
    for area, mw in area_mw.items():
        c = cand[cand.area == area]
        how = "derived:substation_size_line_ratings_within_area"
        if c.empty:
            c = bus[(bus.area == area) & ~bus.is_boundary].sort_values("base_kv").drop_duplicates("node_code")
            how = "derived:any_bus_in_area_line_ratings"
        if c.empty:
            log.warning("planning area %s (%.0f MW) has no bus; load not placed", area, mw)
            rows.append({"bus": None, "area": area, "p_mw": mw, "how": "unplaced"})
            continue
        w = c.node_code.map(node_size).fillna(0).values + 1.0
        for b, wi in zip(c.bus, w / w.sum()):
            rows.append({"bus": int(b), "area": area, "p_mw": mw * wi, "how": how})
    L = pd.DataFrame(rows)
    unplaced = L[L.bus.isna()]
    L = L.dropna(subset=["bus"]).groupby(["bus", "area", "how"], as_index=False).p_mw.sum()
    out = pd.DataFrame({"load": np.arange(1, len(L) + 1)})
    pv.put(out, "bus", L.bus.astype(int).values, L.how.values)
    pv.put(out, "planning_area", L.area.values, "derived:within_aeso_planning_area")
    pv.put(out, "p_mw", L.p_mw.values, f"derived:open_aeso_planning_area_{p['load_column']}_allocated")
    tanphi = math.tan(math.acos(p["load_power_factor"]))
    pv.put(out, "q_mvar", out.p_mw * tanphi, "synthetic:fixed_power_factor")
    out.attrs["unplaced_mw"] = float(unplaced.p_mw.sum())
    out.attrs["area_total_mw"] = float(area_mw.sum())
    return out


# ---------------------------------------------------------------- pandapower net
def to_pandapower(cfg, bus, br, tr, dc, bd, gen, load) -> pp.pandapowerNet:
    o = cfg["opf"]
    net = pp.create_empty_network(f_hz=60.0, sn_mva=o["base_mva"])
    idx = {}
    for r in bus.itertuples():
        idx[r.bus] = pp.create_bus(net, vn_kv=float(r.base_kv), name=r.node_code,
                                   min_vm_pu=o["vmin_pu"], max_vm_pu=o["vmax_pu"],
                                   geodata=(r.lon, r.lat))
    w = 2 * math.pi * 60
    for r in br.itertuples():
        vn = float(r.kv_class)
        pp.create_line_from_parameters(
            net, idx[r.from_bus], idx[r.to_bus], length_km=1.0, r_ohm_per_km=r.r_ohm,
            x_ohm_per_km=r.x_ohm_model, c_nf_per_km=r.b_us * 1e3 / w,
            max_i_ka=r.rate_mva / (math.sqrt(3) * vn), name=str(r.circuit_id), max_loading_percent=100.0)
    for r in tr.itertuples():
        pp.create_transformer_from_parameters(
            net, idx[r.hv_bus], idx[r.lv_bus], sn_mva=r.sn_mva, vn_hv_kv=float(r.hv_kv),
            vn_lv_kv=float(r.lv_kv), vkr_percent=r.vkr_percent, vk_percent=r.vk_percent,
            pfe_kw=0.0, i0_percent=0.0, max_loading_percent=100.0, name=r.node_code)
    tcfg = cfg["topology"]
    for r in dc.itertuples():   # fixed-schedule HVDC (see config topology.hvdc_schedule_mw)
        s = float(np.clip(tcfg["hvdc_schedule_mw"].get(r.name, 0), -r.p_max_mw, r.p_max_mw))
        loss = abs(s) * tcfg["hvdc_loss_percent"] / 100
        send, recv = (r.from_bus, r.to_bus) if s >= 0 else (r.to_bus, r.from_bus)
        pp.create_load(net, idx[send], p_mw=abs(s), q_mvar=0.0, name=f"hvdc_{r.name}_send")
        g = pp.create_gen(net, idx[recv], p_mw=abs(s) - loss, min_p_mw=abs(s) - loss,
                          max_p_mw=abs(s) - loss, min_q_mvar=-0.3 * r.p_max_mw,
                          max_q_mvar=0.3 * r.p_max_mw, controllable=True, name=f"hvdc_{r.name}_recv")
        pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=0.0)
    slack = gen.loc[gen.p_max_mw.idxmax(), "gen"]
    for r in gen.itertuples():
        pmax = float(r.p_max_mw * r.availability)
        g = pp.create_gen(net, idx[r.bus], p_mw=min(pmax, 0.5 * r.p_max_mw), vm_pu=1.0,
                          min_p_mw=float(min(r.p_min_mw, pmax)), max_p_mw=pmax,
                          min_q_mvar=float(r.q_min_mvar), max_q_mvar=float(r.q_max_mvar),
                          controllable=True, slack=bool(r.gen == slack), name=r.unit_code)
        pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=float(r.cost_c1), cp2_eur_per_mw2=float(r.cost_c2))
    sched = cfg["params"]["intertie_schedule_mw"]
    for r in bd.itertuples():
        share = (bd.neighbour == r.neighbour).sum()
        mw = sched.get(r.neighbour, 0) / share
        g = pp.create_gen(net, idx[r.bus], p_mw=mw, min_p_mw=mw, max_p_mw=mw, min_q_mvar=-50,
                          max_q_mvar=50, controllable=True, name=f"intertie_{r.neighbour}_{r.bus}")
        pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=0.0)
    for r in load.itertuples():
        pp.create_load(net, idx[r.bus], p_mw=float(r.p_mw), q_mvar=float(r.q_mvar), controllable=False)
    return net


def build(cfg: dict) -> dict:
    raw = path(cfg, "raw_dir")
    topo = path(cfg, "interim_dir") / "topology"
    out = path(cfg, "interim_dir") / "params"
    out.mkdir(parents=True, exist_ok=True)
    ts, season = _snapshot(cfg)
    bus = pd.read_csv(topo / "buses.csv")
    br = pd.read_csv(topo / "branches.csv")
    tr = pd.read_csv(topo / "transformers.csv")
    dc = pd.read_csv(topo / "dc_lines.csv")
    bd = pd.read_csv(topo / "boundary.csv")
    br = line_params(cfg, br, season)
    tr = transformer_params(cfg, tr, br)
    gen = generators(cfg, raw, bus, ts)
    load = loads(cfg, raw, bus, br, ts)
    for name, df in (("branch_e", br), ("transformer_e", tr), ("generator", gen), ("load", load)):
        pv.validate(df, name)
    br.to_csv(out / "branches.csv", index=False)
    tr.to_csv(out / "transformers.csv", index=False)
    gen.to_csv(out / "generators.csv", index=False)
    load.to_csv(out / "loads.csv", index=False)
    net = to_pandapower(cfg, bus, br, tr, dc, bd, gen, load)
    tag = ts.strftime("%Y%m%d_%H%M")
    pp.to_json(net, out / f"net_{tag}.json")
    summary = {
        "snapshot_mpt": str(ts), "season_ratings": season,
        "generators_active": len(gen), "generators_dropped_inactive": gen.attrs["dropped"],
        "installed_mw_by_fuel": gen.groupby("fuel").p_max_mw.sum().round(0).to_dict(),
        "available_mw_by_fuel": (gen.p_max_mw * gen.availability).groupby(gen.fuel).sum().round(0).to_dict(),
        "load_mw": round(load.p_mw.sum(), 1), "load_area_total_mw": round(load.attrs["area_total_mw"], 1),
        "load_unplaced_mw": round(load.attrs["unplaced_mw"], 1), "load_buses": int(load.bus.nunique()),
        "transformers_upsized": int((tr.sn_mva__src == "derived:sized_to_connected_line_ratings").sum()),
        "x_ohm_per_km_median_by_kv": (br.x_ohm / br.length_km.replace(0, np.nan)).groupby(br.kv_class)
        .median().round(3).to_dict(),
    }
    (out / f"params_summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("params: %s", summary)
    return {"net": net, "summary": summary, "tag": tag}
