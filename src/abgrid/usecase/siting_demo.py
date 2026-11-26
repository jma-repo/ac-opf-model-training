"""Data for the siting / switching visual (all numbers from real OPF solves on
the approximate model; costs illustrative).

1. base case (AC-OPF)
2. single data centre at each hub x size (AC-OPF)
3. buildout: data centres at all hubs at once (AC-OPF)
4. transmission switching on the buildout: greedy DC-OPF screening of single
   line openings (non-bridge lines near congestion), up to K steps, then AC-OPF
   confirmation of the chosen set.
Writes outputs/visual/siting_data.json."""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandapower as pp

from ..config import path
from ..solve.opf import ac_opf, dc_opf

log = logging.getLogger("abgrid.siting")

HUBS = {"Calgary": (51.05, -114.07, 240), "Edmonton": (53.55, -113.49, 240),
        "Genesee": (53.34, -114.30, 240), "Red Deer": (52.27, -113.81, 240),
        "Grande Prairie": (55.17, -118.80, 138), "Fort McMurray": (56.73, -111.38, 240),
        "Lethbridge": (49.69, -112.84, 240), "Medicine Hat": (50.04, -110.68, 240),
        "Brooks": (50.56, -111.90, 240)}
BUILDOUT = {"Calgary": 500, "Edmonton": 500, "Genesee": 500, "Red Deer": 300, "Grande Prairie": 300,
            "Fort McMurray": 300, "Lethbridge": 300, "Medicine Hat": 100, "Brooks": 100}
SIZES = [100, 300, 500]
PF = 0.98


def _hub_bus(net, la, lo, kv_min):
    b = net.bus[net.bus.vn_kv >= kv_min]
    xy = np.array([g for g in b.geo.map(lambda s: json.loads(s)["coordinates"])])
    d = np.hypot((xy[:, 1] - la) * 111, (xy[:, 0] - lo) * 111 * math.cos(math.radians(la)))
    return int(b.index[int(np.argmin(d))]), float(d.min())


def _add_dc(net, bus, mw, tag):
    pp.create_load(net, bus, p_mw=mw, q_mvar=mw * math.tan(math.acos(PF)), name=f"datacentre_{tag}")


def economics(net) -> dict:
    """Hourly settlement decomposition (CAD/h, illustrative costs):
    load payment = sum LMP*Pd; generator revenue = sum LMP*Pg;
    fuel = OPF cost; generator margin = revenue - fuel;
    congestion & loss rent = payment - revenue."""
    lam = net.res_bus.lam_p
    ld = net.load[net.load.in_service]
    pay = float((lam[ld.bus].values * net.res_load.p_mw[ld.index].values).sum())
    g = net.gen[net.gen.in_service]
    rev = float((lam[g.bus].values * net.res_gen.p_mw[g.index].values).sum())
    fuel = float(net.res_cost)
    ren = g.name.astype(str).str.match(r"AB_.*") & g.index.isin(g.index)
    return {"customer_payment": round(pay, 0), "fuel_cost": round(fuel, 0),
            "generator_margin": round(rev - fuel, 0), "congestion_loss_rent": round(pay - rev, 0)}


def curtailment(net, fuels) -> float:
    g = net.gen[net.gen.in_service & net.gen.name.isin(fuels.index)]
    ren = g[fuels[g.name].isin(["wind", "solar"]).values]
    return round(float((ren.max_p_mw - net.res_gen.p_mw[ren.index]).clip(lower=0).sum()), 1)


def snapshot(net, rec, fuels, keep_fields=True) -> dict:
    out = {"feasible": rec["feasible"], "relaxation_level": rec["relaxation_level"],
           "relaxation_name": rec.get("relaxation_name")}
    if rec["feasible"]:
        hv = net.load.name.astype(str).str.startswith("hvdc_")
        out.update(economics(net))
        out["load_mw"] = round(float(net.res_load.p_mw[~hv].sum()), 0)
        out["curtailed_mw"] = curtailment(net, fuels)
        out["lmp"] = [round(float(x), 1) for x in net.res_bus.lam_p.values]
        r, L = net.res_line, net.line
        rate = np.sqrt(3) * L.max_i_ka * net.bus.vn_kv[L.from_bus].values
        s = np.maximum(np.hypot(r.p_from_mw, r.q_from_mvar), np.hypot(r.p_to_mw, r.q_to_mvar))
        out["loading"] = [int(round(x)) if np.isfinite(x) else 0 for x in (100 * s / rate)]
        # +1: active power flows from_bus -> to_bus, -1: reverse, 0: out of service
        out["flow_dir"] = [0 if not ins else (1 if p >= 0 else -1)
                           for p, ins in zip(r.p_from_mw.fillna(0), L.in_service)]
        out["flow_mw"] = [round(float(abs(p)), 0) if np.isfinite(p) else 0 for p in r.p_from_mw]
        out["lmp_median"] = round(float(np.median(net.res_bus.lam_p)), 1)
        out["lmp_max"] = round(float(net.res_bus.lam_p.max()), 1)
    return out


def _non_bridge_lines(net):
    G = nx.MultiGraph()
    G.add_edges_from((r.from_bus, r.to_bus, ("l", i)) for i, r in net.line[net.line.in_service].iterrows())
    G.add_edges_from((r.hv_bus, r.lv_bus, ("t", i)) for i, r in net.trafo.iterrows())
    br = {frozenset(e) for e in nx.bridges(nx.Graph(G))}
    pair = {i: frozenset((r.from_bus, r.to_bus)) for i, r in net.line.iterrows()}
    cnt = {}
    for p in pair.values():
        cnt[p] = cnt.get(p, 0) + 1
    return {i for i in net.line.index[net.line.in_service] if pair[i] not in br or cnt[pair[i]] > 1}


def _shed_mw(sol):
    hv = sol.load.name.astype(str).str.startswith("hvdc_")
    return float((sol.load.p_mw[~hv] - sol.res_load.p_mw[~hv]).sum())


def switching(cfg, net, k_steps=3, n_cand=50):
    """Greedy optimal-transmission-switching heuristic (DC screening).
    Loads may be shed at VOLL in the screening model, so an infeasible
    buildout still gets a finite score; an opening is kept if it lowers
    cost + VOLL x shed. Nominal line ratings are used throughout."""
    from ..solve.opf import _apply
    lvl1 = [r for r in cfg["opf"]["relaxations"] if "rating_factor" in r][0]
    opened, steps = [], []
    cur = copy.deepcopy(net)
    nominal_ok = dc_opf(cur)[1]
    if not nominal_ok:      # screen on the level-1 model (ratings x factor)
        _apply(cur, lvl1, cfg)
    sol, ok, _ = dc_opf(cur)
    if not ok:
        return opened, steps
    best_cost = float(sol.res_cost)
    steps.append({"step": 0, "dc_objective": round(best_cost, 0), "opened": None,
                  "screening_model": "nominal" if nominal_ok else f"ratings x{lvl1['rating_factor']}",
                  "dc_feasible_nominal": nominal_ok})
    for k in range(1, k_steps + 1):
        load = sol.res_line.loading_percent
        hot = set(load[load > 95].index)
        near = set()
        hb = set(cur.line.from_bus[list(hot)]) | set(cur.line.to_bus[list(hot)])
        near = set(cur.line.index[cur.line.from_bus.isin(hb) | cur.line.to_bus.isin(hb)])
        cand = [i for i in (hot | near) if i in _non_bridge_lines(cur)]
        cand = sorted(cand, key=lambda i: -load.get(i, 0))[:n_cand]
        trial = []
        for i in cand:
            t = copy.deepcopy(cur)
            t.line.loc[i, "in_service"] = False
            s, ok2, _ = dc_opf(t)
            if ok2:
                trial.append((float(s.res_cost), i, s))
        if not trial:
            break
        c, i, s = min(trial, key=lambda x: x[0])
        if c >= best_cost - 1.0:
            break
        cur.line.loc[i, "in_service"] = False
        opened.append(int(i)); best_cost, sol = c, s
        chk = copy.deepcopy(net)
        chk.line.loc[opened, "in_service"] = False
        nom = dc_opf(chk)[1]
        steps.append({"step": k, "dc_objective": round(c, 0), "opened": str(cur.line.name[i]),
                      "line_index": int(i), "screened": len(trial), "dc_feasible_nominal": nom})
        log.info("switching step %d: open %s, DC objective %.0f, nominal-feasible=%s (%d screened)",
                 k, cur.line.name[i], c, nom, len(trial))
    return opened, steps


def run_switching(cfg):
    """Recompute buildout switching only, reusing saved single-site results."""
    import pandas as pd
    out = path(cfg, "output_dir") / "visual"
    res = json.loads((out / "siting_data.json").read_text())
    pdir = path(cfg, "interim_dir") / "params"
    net0 = pp.from_json(sorted(pdir.glob("net_*.json"))[-1])
    fuels = pd.read_csv(pdir / "generators.csv").set_index("unit_code").fuel
    nb = copy.deepcopy(net0)
    for h, mw in BUILDOUT.items():
        _add_dc(nb, res["hubs"][h]["bus"], mw, h)
    opened, steps = switching(cfg, nb)
    res["switching_steps"] = steps
    ns = copy.deepcopy(nb)
    ns.line.loc[opened, "in_service"] = False
    ss, rs = ac_opf(ns, cfg)
    res["switched"] = {"opened": [int(i) for i in opened],
                       "opened_names": [str(ns.line.name[i]) for i in opened], **snapshot(ss, rs, fuels)}
    log.info("switched AC: feasible=%s level=%s", rs["feasible"], rs["relaxation_level"])
    buses = pd.read_csv(path(cfg, "interim_dir") / "topology" / "buses.csv")
    res["geo"]["bus_name"] = buses.name.tolist()
    (out / "siting_data.json").write_text(json.dumps(res))
    return res


def run(cfg):
    out = path(cfg, "output_dir") / "visual"
    out.mkdir(parents=True, exist_ok=True)
    pdir = path(cfg, "interim_dir") / "params"
    net0 = pp.from_json(sorted(pdir.glob("net_*.json"))[-1])
    import pandas as pd
    fuels = pd.read_csv(pdir / "generators.csv").set_index("unit_code").fuel

    hubs = {}
    for h, (la, lo, kv) in HUBS.items():
        b, d = _hub_bus(net0, la, lo, kv)
        hubs[h] = {"bus": b, "node": net0.bus.name[b], "kv": float(net0.bus.vn_kv[b]), "dist_km": round(d, 1)}

    res = {"meta": {"snapshot_mpt": cfg["params"]["snapshot_mpt"], "pf": PF,
                    "note": "Approximate open-data Alberta model, NOT AESO's grid. Costs illustrative."}}
    b0, r0 = ac_opf(net0, cfg)
    res["base"] = snapshot(b0, r0, fuels)
    log.info("base: %s", {k: v for k, v in res["base"].items() if k not in ("lmp", "loading")})

    single = {}
    for h, hb in hubs.items():
        for mw in SIZES:
            n = copy.deepcopy(net0)
            _add_dc(n, hb["bus"], mw, h)
            s, r = ac_opf(n, cfg)
            snap = snapshot(s, r, fuels)
            single[f"{h}|{mw}"] = {k: v for k, v in snap.items() if k not in ("loading",)}
            log.info("%s %d MW: feasible=%s level=%s pay=%s", h, mw, r["feasible"], r["relaxation_level"],
                     snap.get("customer_payment"))
    res["single"] = single

    nb = copy.deepcopy(net0)
    for h, mw in BUILDOUT.items():
        _add_dc(nb, hubs[h]["bus"], mw, h)
    sb, rb = ac_opf(nb, cfg)
    res["buildout"] = {"sizes": BUILDOUT, **snapshot(sb, rb, fuels)}
    log.info("buildout: feasible=%s level=%s", rb["feasible"], rb["relaxation_level"])

    opened, steps = switching(cfg, nb)
    res["switching_steps"] = steps
    ns = copy.deepcopy(nb)
    ns.line.loc[opened, "in_service"] = False
    ss, rs = ac_opf(ns, cfg)
    res["switched"] = {"opened": [int(i) for i in opened],
                       "opened_names": [str(ns.line.name[i]) for i in opened], **snapshot(ss, rs, fuels)}
    log.info("switched AC: feasible=%s level=%s", rs["feasible"], rs["relaxation_level"])

    # geometry: buses and simplified branch polylines (ordered as net.line)
    topo = path(cfg, "interim_dir") / "topology"
    brg = gpd.read_file(topo / "branches.geojson")
    geo = [[[round(x, 3), round(y, 3)] for x, y in g.simplify(0.01).coords] for g in brg.geometry]
    bxy = [json.loads(s)["coordinates"] for s in net0.bus.geo]
    res["geo"] = {"bus": [[round(x, 3), round(y, 3)] for x, y in bxy],
                  "bus_kv": net0.bus.vn_kv.astype(int).tolist(),
                  "line": geo, "line_kv": [int(v) for v in brg.kv_class],
                  "line_name": [str(v) for v in brg.circuit_id],
                  "trafo": [[int(a), int(b)] for a, b in zip(net0.trafo.hv_bus, net0.trafo.lv_bus)]}
    res["hubs"] = hubs
    (out / "siting_data.json").write_text(json.dumps(res))
    return res
