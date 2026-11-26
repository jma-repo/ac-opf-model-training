"""Phase 5 scenario generator.

Each scenario = one real 2024 hour (AESO planning-area load + metered
wind/solar output) plus a perturbation family:
  base_hourly | n1_line | n1_gen | large_load (data-centre-sized load).
HVDC and intertie schedules are sampled (synthetic). Every scenario is solved
with DC-OPF and AC-OPF (relaxation hierarchy); feasible and infeasible cases
are both kept with labels. Specs are sampled up front from one seed, so a run
is reproducible from config alone."""
from __future__ import annotations

import copy
import json
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp

from ..config import path
from ..export import case as ex
from ..solve.opf import ac_opf, dc_opf, sanity

log = logging.getLogger("abgrid.scenarios")


# ------------------------------------------------------------------ inputs
def _inputs(cfg):
    raw = path(cfg, "raw_dir")
    pdir = path(cfg, "interim_dir") / "params"
    yr = cfg["scenarios"]["hours_year"]
    pa = pd.read_csv(next((raw / "aeso_files").glob(f"Planning-Area-Load-Hourly-{yr}/*.csv")),
                     usecols=["DT_MPT", "PLANNING_AREA", cfg["params"]["load_column"]])
    pa = pa.groupby(["DT_MPT", "PLANNING_AREA"])[cfg["params"]["load_column"]].sum().unstack()
    pa.index = pd.to_datetime(pa.index)
    mv = pd.concat([pd.read_csv(f) for f in sorted((raw / "aeso").glob(f"metered_*_{yr}-01-01_{yr}-12-31.csv.gz"))])
    mv["t"] = pd.to_datetime(mv.begin_date_mpt)
    mv = mv.pivot_table(index="t", columns="asset_ID", values="metered_volume_mwh", aggfunc="sum")
    hours = pa.index.intersection(mv.index)
    gens = pd.read_csv(pdir / "generators.csv")
    loads = pd.read_csv(pdir / "loads.csv")
    cal = pdir / "calibration.json"
    return {"pa": pa.loc[hours], "mv": mv.loc[hours].clip(lower=0), "hours": hours, "gens": gens, "loads": loads,
            "net": sorted(pdir.glob("net_*.json"))[-1], "umap": unit_asset_map(cfg, gens),
            "cal": json.loads(cal.read_text()) if cal.exists() else None}


def unit_asset_map(cfg, gens) -> pd.DataFrame:
    """CODERS unit -> (AESO metered asset, share of that asset's output).
    Default: unit-code pattern; overrides for facility-level metering, with
    output split across the facility's units by installed capacity."""
    ov = {u: a for a, us in cfg.get("calibration", {}).get("aeso_asset_overrides", {}).items() for u in us}
    g = gens[["unit_code", "aeso_asset_id", "p_max_mw", "fuel"]].copy()
    g["asset"] = [ov.get(a, a) for a in g.aeso_asset_id]
    g["share"] = g.p_max_mw / g.groupby("asset").p_max_mw.transform("sum")
    g["src"] = np.where(g.aeso_asset_id.isin(ov), "derived:facility_metering_capacity_share",
                        "derived:coders_unit_code_pattern")
    return g.set_index("unit_code")


def unit_actual(inp, h) -> pd.Series:
    """Metered MW per CODERS unit at hour h (NaN where no metering)."""
    um, row = inp["umap"], inp["mv"].loc[h]
    return um.asset.map(row) * um.share


def set_costs(net, c1_by_unit: dict, c2: float):
    pc = net.poly_cost
    gi = pc.et == "gen"
    names = net.gen.name.reindex(pc.element[gi].values).values
    for k, (idx, nm) in enumerate(zip(pc.index[gi], names)):
        if nm in c1_by_unit:
            pc.loc[idx, "cp1_eur_per_mw"] = c1_by_unit[nm]
            pc.loc[idx, "cp2_eur_per_mw2"] = c2


def sample_specs(cfg, inp) -> list[dict]:
    s = cfg["scenarios"]
    rng = np.random.default_rng(s["seed"])
    net = pp.from_json(inp["net"])
    fam = list(s["families"])
    probs = np.array([s["families"][f] for f in fam], float)
    counts = np.floor(probs / probs.sum() * s["n"]).astype(int)
    counts[0] += s["n"] - counts.sum()
    # non-bridge AC branches (an outage keeps one island)
    G = nx.MultiGraph()
    G.add_edges_from((r.from_bus, r.to_bus, ("line", i)) for i, r in net.line.iterrows())
    G.add_edges_from((r.hv_bus, r.lv_bus, ("trafo", i)) for i, r in net.trafo.iterrows())
    simple = nx.Graph(G)
    bridges = {frozenset(e) for e in nx.bridges(simple)}
    par = pd.Series([frozenset((r.from_bus, r.to_bus)) for _, r in net.line.iterrows()], index=net.line.index)
    multi = par.map(par.value_counts())
    outage_lines = [i for i in net.line.index if par[i] not in bridges or multi[i] > 1]
    real_gens = [i for i in net.gen.index
                 if not str(net.gen.name[i]).startswith(("hvdc_", "intertie_")) and not net.gen.slack[i]]
    gw = net.gen.max_p_mw[real_gens].clip(lower=1).values
    cand = net.bus.index[net.bus.vn_kv.isin(s["large_load"]["candidate_kv"])
                         & ~net.bus.name.str.contains("IPT|INT")].tolist()
    specs, k = [], 0
    for f, c in zip(fam, counts):
        for _ in range(c):
            h = inp["hours"][rng.integers(len(inp["hours"]))]
            sp = {"id": k, "family": f, "hour": str(h),
                  "hvdc_mw": {n: float(rng.uniform(*s["hvdc_schedule_range_mw"]))
                              for n in cfg["topology"]["hvdc_schedule_mw"]},
                  "intertie_u": {n: float(rng.uniform()) for n in cfg["topology"]["boundary"]},
                  "noise_seed": int(rng.integers(2**31))}
            if f == "n1_line":
                sp["line_out"] = int(rng.choice(outage_lines))
            elif f == "n1_gen":
                sp["gen_out"] = int(rng.choice(real_gens, p=gw / gw.sum()))
            elif f == "large_load":
                sp["dc_bus"] = int(rng.choice(cand))
                sp["dc_mw"] = float(rng.choice(s["large_load"]["sizes_mw"]))
            specs.append(sp)
            k += 1
    rng.shuffle(specs)
    return specs


def _tie_rating(n, bus) -> float:
    L = n.line[(n.line.from_bus == bus) | (n.line.to_bus == bus)]
    return float((np.sqrt(3) * L.max_i_ka * n.bus.vn_kv[bus]).sum())


# ------------------------------------------------------------------ apply one spec
def apply_spec(cfg, base, inp, sp) -> pp.pandapowerNet:
    s = cfg["scenarios"]
    n = copy.deepcopy(base)
    h = pd.Timestamp(sp["hour"])
    rng = np.random.default_rng(sp["noise_seed"])
    # loads: AESO planning-area load at hour h, same within-area weights as base
    L = inp["loads"]
    is_hv = n.load.name.astype(str).str.startswith("hvdc_")
    idx = n.load.index[~is_hv]
    area_base = L.groupby("planning_area").p_mw.transform("sum").values
    w = L.p_mw.values / area_base
    area_now = inp["pa"].loc[h].reindex(L.planning_area.values).fillna(0).values
    sig = sp.get("noise", s["local_load_noise"])
    eps = rng.uniform(-sig, sig, len(idx))
    p = w * area_now * (1 + eps)
    tanphi = math.tan(math.acos(cfg["params"]["load_power_factor"]))
    n.load.loc[idx, "p_mw"] = p
    n.load.loc[idx, "q_mvar"] = p * tanphi
    # metered output at hour h: wind/solar set the available maximum; fixed-output
    # fuels (cogeneration, hydro, biomass) are pinned to it. Units without metering
    # use the fleet ratio of metered units of the same fuel.
    G = inp["gens"].set_index("unit_code")
    act = unit_actual(inp, h)
    fixed = set(cfg.get("calibration", {}).get("fixed_output_fuels", []))
    name_to_gi = pd.Series(n.gen.index, index=n.gen.name.values)
    for f in sorted({"wind", "solar"} | fixed):
        units = G[G.fuel == f]
        a = act.reindex(units.index)
        known = a.notna()
        ratio = float(a[known].sum() / max(units.p_max_mw[known].sum(), 1)) if known.any() else 0.0
        mw = (a.where(known, ratio * units.p_max_mw)).clip(lower=0).clip(upper=units.p_max_mw)
        gi = name_to_gi.reindex(units.index).dropna().astype(int)
        mw = mw.reindex(gi.index).values
        n.gen.loc[gi.values, "max_p_mw"] = mw
        if f in fixed:
            n.gen.loc[gi.values, "min_p_mw"] = mw
            n.gen.loc[gi.values, "p_mw"] = mw
        else:
            n.gen.loc[gi.values, "min_p_mw"] = 0.0
            n.gen.loc[gi.values, "p_mw"] = 0.5 * mw
    if inp.get("cal") and inp["cal"].get("use_costs", True):
        set_costs(n, inp["cal"]["c1_by_unit"], inp["cal"]["c2"])
    # HVDC fixed schedules
    loss = cfg["topology"]["hvdc_loss_percent"] / 100
    for name, mw in sp["hvdc_mw"].items():
        n.load.loc[n.load.name == f"hvdc_{name}_send", "p_mw"] = mw
        gi = n.gen.index[n.gen.name == f"hvdc_{name}_recv"]
        n.gen.loc[gi, ["p_mw", "min_p_mw", "max_p_mw"]] = mw * (1 - loss)
    # interties: uniform in [-export, import] per neighbour, capped at 90% of the
    # summed tie-line ratings and split over tie buses in proportion to rating
    if "net_import_mw" in sp:        # observed total interchange, split by tie rating
        gi = n.gen.index[n.gen.name.astype(str).str.startswith("intertie_")]
        rate = np.array([_tie_rating(n, n.gen.bus[g]) for g in gi])
        mw = float(np.clip(sp["net_import_mw"], -0.9 * rate.sum(), 0.9 * rate.sum()))
        for g, r in zip(gi, rate):
            n.gen.loc[g, ["p_mw", "min_p_mw", "max_p_mw"]] = mw * r / rate.sum()
    else:
      for nb, b in cfg["topology"]["boundary"].items():
        gi = n.gen.index[n.gen.name.astype(str).str.startswith(f"intertie_{nb}_")]
        if not len(gi):
            continue
        rate = np.array([_tie_rating(n, n.gen.bus[g]) for g in gi])
        cap = 0.9 * rate.sum()
        lo, hi = -min(b["export_max_mw"], cap), min(b["import_max_mw"], cap)
        mw = lo + sp["intertie_u"][nb] * (hi - lo)
        for g, r in zip(gi, rate):
            n.gen.loc[g, ["p_mw", "min_p_mw", "max_p_mw"]] = mw * r / rate.sum()
    if "line_out" in sp:
        n.line.loc[sp["line_out"], "in_service"] = False
    if "gen_out" in sp:
        n.gen.loc[sp["gen_out"], "in_service"] = False
    if "dc_bus" in sp:
        pf = s["large_load"]["power_factor"]
        pp.create_load(n, sp["dc_bus"], p_mw=sp["dc_mw"], q_mvar=sp["dc_mw"] * math.tan(math.acos(pf)),
                       name="datacentre")
    return n


# ------------------------------------------------------------------ worker
_W: dict = {}


def _init(cfg, out_dir):
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    inp = _inputs(cfg)
    _W.update(cfg=cfg, inp=inp, base=pp.from_json(inp["net"]), out=Path(out_dir))


def _run(sp):
    cfg, inp, out = _W["cfg"], _W["inp"], _W["out"]
    t0 = time.perf_counter()
    n = apply_spec(cfg, _W["base"], inp, sp)
    rec = {**{k: v for k, v in sp.items() if k not in ("hvdc_mw", "intertie_u")},
           "hvdc_watl_mw": sp["hvdc_mw"].get("WATL"), "hvdc_eatl_mw": sp["hvdc_mw"].get("EATL"),
           "load_mw": float(n.load.p_mw[~n.load.name.astype(str).str.startswith("hvdc_")].sum()),
           "avail_mw": float(n.gen.max_p_mw[n.gen.in_service].sum())}
    ndc, dc_ok, dc_t = dc_opf(n)
    rec.update(dc_feasible=dc_ok, dc_s=round(dc_t, 3),
               dc_cost=float(ndc.res_cost) if dc_ok else None)
    if dc_ok:
        n.gen.p_mw = ndc.res_gen.p_mw.values
    if cfg["scenarios"]["relax_in_scenarios"]:
        sol, r = ac_opf(n, cfg)
    else:
        sol, r = ac_opf(n, {**cfg, "opf": {**cfg["opf"], "relaxations": []}})
    lvl = r["relaxation_level"]
    rec.update(ac_feasible_any=r["feasible"], relaxation_level=lvl,
               feasible=bool(r["feasible"] and lvl == 0),
               ac_s_nominal=r["attempts"][0]["seconds"],
               ac_s_total=round(sum(a["seconds"] for a in r["attempts"]), 2))
    if r["feasible"]:
        sc = sanity(sol, ac=True)
        rec.update(cost=sc["cost"], losses_pct=sc["losses_%"], vm_min=sc["vm_min"], vm_max=sc["vm_max"],
                   lmp_median=sc["lmp_median"], load_shed_mw=sc["load_shed_mw"])
    if cfg["scenarios"]["export_pyg"]:
        meta = {"scenario_id": sp["id"], "family": sp["family"], "hour_mpt": sp["hour"],
                "relaxation_level": lvl, "feasible": rec["feasible"]}
        # nominal-feasible cases carry a solution; others are written unsolved, labelled
        net_out = sol if rec["feasible"] else n
        d = ex.to_gridsfm(ex.build_case(net_out, solved=rec["feasible"]), meta)
        d["metadata"]["feasible"] = rec["feasible"]
        (out / "pyg" / f"scn_{sp['id']:05d}.pyg.json").write_text(json.dumps(d, default=ex._json_default))
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    return rec


def generate(cfg, n: int | None = None, run_name: str | None = None, n_proc: int | None = None):
    s = cfg["scenarios"]
    run = run_name or s["run_name"]
    out = path(cfg, "interim_dir") / "scenarios" / run
    (out / "pyg").mkdir(parents=True, exist_ok=True)
    inp = _inputs(cfg)
    specs = sample_specs(cfg, inp)
    if n is not None:
        specs = specs[:n]
    (out / "specs.jsonl").write_text("\n".join(json.dumps(x) for x in specs))
    done = set()
    lab = out / "labels.jsonl"
    if lab.exists():   # resume
        done = {json.loads(line)["id"] for line in lab.read_text().splitlines() if line.strip()}
    todo = [x for x in specs if x["id"] not in done]
    log.info("run %s: %d specs, %d to do, %d processes", run, len(specs), len(todo), n_proc or s["n_processes"])
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(n_proc or s["n_processes"], initializer=_init,
                                       initargs=(cfg, str(out))) as pool, open(lab, "a") as fh:
        for i, rec in enumerate(pool.imap_unordered(_run, todo, chunksize=1), 1):
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log.info("%d/%d done, %.1f s elapsed, %.2f s/scenario wall", i, len(todo), el, el / i)
    df = pd.DataFrame([json.loads(x) for x in lab.read_text().splitlines() if x.strip()])
    df.to_csv(out / "labels.csv", index=False)
    return df, out
