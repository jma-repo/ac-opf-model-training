"""Convert a pandapower net (optionally with an AC-OPF solution) into the
GridSFM / OPFData `.pyg.json` layout. Everything in per-unit on base_mva.

Column layouts (gridsfm/schema.py, OPFData widths 4/11/9/11):
 bus         [base_kV, type, Vmin, Vmax]
 generator   [mbase, pg, Pmin, Pmax, qg, Qmin, Qmax, Vg, cp2, cp1, cp0]
 load        [Pd, Qd]
 ac_line     [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
 transformer [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
 solution    bus [va(rad), vm]; generator [pg, qg]; edges [pf, qf, pt, qt]
Indices are 0-based. Costs: cp1 = c1*base, cp2 = c2*base^2 (per-unit power).
"""
from __future__ import annotations

import json
import math

import numpy as np

ANG = math.radians(30.0)


def convert(net, solved: bool, meta: dict | None = None) -> dict:
    base = float(net.sn_mva)
    bus_ix = {b: i for i, b in enumerate(net.bus.index)}
    slack_bus = set(net.gen.bus[net.gen.slack]) | set(net.ext_grid.bus)
    gen_bus = set(net.gen.bus)
    bus = [[float(r.vn_kv), 3 if b in slack_bus else 2 if b in gen_bus else 1,
            float(r.min_vm_pu), float(r.max_vm_pu)] for b, r in net.bus.iterrows()]

    cost = net.poly_cost[net.poly_cost.et == "gen"].set_index("element")
    gens, glink = [], []
    for k, (g, r) in enumerate(net.gen.iterrows()):
        c = cost.loc[g] if g in cost.index else None
        pg = float(net.res_gen.p_mw[g]) if solved else float(r.p_mw)
        qg = float(net.res_gen.q_mvar[g]) if solved else 0.0
        gens.append([max(float(r.max_p_mw), 1.0), pg / base, float(r.min_p_mw) / base,
                     float(r.max_p_mw) / base, qg / base, float(r.min_q_mvar) / base,
                     float(r.max_q_mvar) / base, float(r.vm_pu),
                     float(c.cp2_eur_per_mw2) * base ** 2 if c is not None else 0.0,
                     float(c.cp1_eur_per_mw) * base if c is not None else 0.0,
                     float(c.cp0_eur) if c is not None else 0.0])
        glink.append(bus_ix[r.bus])
    loads = [[float(r.p_mw) / base, float(r.q_mvar) / base] for _, r in net.load.iterrows()]
    llink = [bus_ix[b] for b in net.load.bus]

    w = 2 * math.pi * float(net.f_hz)
    lf, ls, lr = [], [], []
    for _, r in net.line.iterrows():
        vn = float(net.bus.vn_kv[r.from_bus])
        zb = vn ** 2 / base
        R = r.r_ohm_per_km * r.length_km / zb
        X = r.x_ohm_per_km * r.length_km / zb
        B = w * r.c_nf_per_km * 1e-9 * r.length_km * zb
        rate = math.sqrt(3) * r.max_i_ka * vn / base
        lf.append([-ANG, ANG, B / 2, B / 2, R, X, rate, rate, rate])
        ls.append(bus_ix[r.from_bus]); lr.append(bus_ix[r.to_bus])
    tf, ts_, tr_ = [], [], []
    for _, r in net.trafo.iterrows():
        z = r.vk_percent / 100 * base / r.sn_mva
        R = r.vkr_percent / 100 * base / r.sn_mva
        X = math.sqrt(max(z ** 2 - R ** 2, 1e-12))
        rate = r.sn_mva / base
        tf.append([-ANG, ANG, R, X, rate, rate, rate, 1.0, 0.0, 0.0, 0.0])
        ts_.append(bus_ix[r.hv_bus]); tr_.append(bus_ix[r.lv_bus])

    out = {"grid": {
        "nodes": {"bus": bus, "generator": gens, "load": loads, "shunt": []},
        "edges": {
            "ac_line": {"senders": ls, "receivers": lr, "features": lf},
            "transformer": {"senders": ts_, "receivers": tr_, "features": tf},
            "generator_link": {"senders": list(range(len(gens))), "receivers": glink},
            "load_link": {"senders": list(range(len(loads))), "receivers": llink},
            "shunt_link": {"senders": [], "receivers": []},
        },
        "context": [[[base]]],
    }}
    if solved:
        rb, rl, rt = net.res_bus, net.res_line, net.res_trafo
        out["solution"] = {
            "nodes": {"bus": [[math.radians(float(rb.va_degree[b])), float(rb.vm_pu[b])]
                              for b in net.bus.index],
                      "generator": [[float(net.res_gen.p_mw[g]) / base, float(net.res_gen.q_mvar[g]) / base]
                                    for g in net.gen.index]},
            "edges": {
                "ac_line": {"senders": ls, "receivers": lr, "features": (np.c_[
                    rl.p_from_mw, rl.q_from_mvar, rl.p_to_mw, rl.q_to_mvar] / base).tolist()},
                "transformer": {"senders": ts_, "receivers": tr_, "features": (np.c_[
                    rt.p_hv_mw, rt.q_hv_mvar, rt.p_lv_mw, rt.q_lv_mvar] / base).tolist()},
            },
        }
    out["metadata"] = {"objective": float(net.res_cost) if solved else None,
                       "feasible": bool(solved), "base_mva": base,
                       "bus_names": net.bus.name.tolist(), "gen_names": net.gen.name.tolist(),
                       "disclaimer": "Approximate open-data Alberta model; NOT AESO's grid.",
                       **(meta or {})}
    return out


def write(net, path, solved: bool, meta=None):
    with open(path, "w") as f:
        json.dump(convert(net, solved, meta), f)
