"""DC-OPF then AC-OPF (pandapower / PYPOWER interior point). If AC-OPF is
infeasible, the documented relaxation hierarchy in config `opf.relaxations`
is applied cumulatively, level by level; the level reached is recorded as a
quality signal. Parameters are never changed silently: every relaxation is
written to the result record."""
from __future__ import annotations

import copy
import json
import logging
import time

import numpy as np
import pandapower as pp

log = logging.getLogger("abgrid.solve")


def _apply(net, relax: dict, cfg):
    if "rating_factor" in relax:
        net.line.max_loading_percent *= relax["rating_factor"]
        net.trafo.max_loading_percent *= relax["rating_factor"]
    if "vmin_pu" in relax:
        net.bus.min_vm_pu = relax["vmin_pu"]
        net.bus.max_vm_pu = relax["vmax_pu"]
    if "svc_mvar" in relax:
        for b in net.bus.index[net.bus.vn_kv >= relax["svc_kv_min"]]:
            g = pp.create_gen(net, b, p_mw=0, min_p_mw=0, max_p_mw=0, vm_pu=1.0,
                              min_q_mvar=-relax["svc_mvar"], max_q_mvar=relax["svc_mvar"],
                              controllable=True, name=f"relax_svc_{b}")
            pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=0.0)
    if "voll" in relax:
        net.load["controllable"] = True
        net.load["min_p_mw"] = 0.0
        net.load["max_p_mw"] = net.load.p_mw
        net.load["min_q_mvar"] = net.load.q_mvar
        net.load["max_q_mvar"] = net.load.q_mvar
        for i in net.load.index:
            pp.create_poly_cost(net, i, "load", cp1_eur_per_mw=-relax["voll"])


def dc_opf(net):
    n = copy.deepcopy(net)
    t = time.perf_counter()
    try:
        pp.rundcopp(n, verbose=False)
        ok = bool(n.OPF_converged)
    except Exception as e:  # noqa: BLE001
        log.warning("DC-OPF failed: %s", e)
        ok = False
    return n, ok, time.perf_counter() - t


def ac_opf(net, cfg, init="pf"):
    """Returns (net, record). Tries nominal, then cumulative relaxations."""
    levels = [{"level": 0, "name": "nominal"}] + list(cfg["opf"]["relaxations"])
    n = copy.deepcopy(net)
    tried = []
    for lv in levels:
        if lv["level"] > 0:
            _apply(n, lv, cfg)
        t = time.perf_counter()
        ok, err = False, None
        for ini in (init, "flat"):
            try:
                pp.runopp(n, init=ini, verbose=False, calculate_voltage_angles=True,
                          numba=False, OPF_FLOW_LIM="S")
                ok = bool(n.OPF_converged)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                ok = False
            if ok:
                break
        dt = time.perf_counter() - t
        tried.append({"level": lv["level"], "name": lv["name"], "converged": ok,
                      "seconds": round(dt, 2), "error": err})
        log.info("AC-OPF level %d (%s): %s in %.1fs", lv["level"], lv["name"], ok, dt)
        if ok:
            return n, {"feasible": True, "relaxation_level": lv["level"],
                       "relaxation_name": lv["name"], "attempts": tried}
    return n, {"feasible": False, "relaxation_level": None, "attempts": tried}


def _s_loading(net):
    """Apparent-power loading vs the MVA rating at nominal voltage (the
    quantity the OPF constrains). pandapower's loading_percent is
    current-based and exceeds 100% when V < 1 pu."""
    r, L = net.res_line, net.line
    rate = np.sqrt(3) * L.max_i_ka * net.bus.vn_kv[L.from_bus].values
    s = np.maximum(np.hypot(r.p_from_mw, r.q_from_mvar), np.hypot(r.p_to_mw, r.q_to_mvar))
    return 100 * s / rate


def sanity(net, ac: bool) -> dict:
    rg, rl = net.res_gen, net.res_load
    hv_g = net.gen.name.astype(str).str.startswith("hvdc_")
    hv_l = net.load.name.astype(str).str.startswith("hvdc_")
    dcl = float(rl.p_mw[hv_l].sum() - rg.p_mw[hv_g].sum())          # HVDC losses
    gen_p = float(rg.p_mw[~hv_g].sum())
    load_p = float(rl.p_mw[~hv_l].sum()) if len(rl) else float(net.load.p_mw[~hv_l].sum())
    gen_p -= dcl                                                     # AC losses only below
    out = {"gen_mw": round(gen_p, 1), "load_mw": round(load_p, 1),
           "losses_mw": round(gen_p - load_p, 1),
           "losses_%": round(100 * (gen_p - load_p) / max(load_p, 1), 2),
           "cost": round(float(net.res_cost), 1),
           "max_line_loading_%": round(float(_s_loading(net).max()), 1),
           "lines_over_100%": int((_s_loading(net) > 100.5).sum()),
           "max_trafo_loading_%": round(float(net.res_trafo.loading_percent.max()), 1),
           "hvdc_mw": [round(x, 1) for x in rl.p_mw[hv_l]],
           "hvdc_loss_mw": round(dcl, 1)}
    if ac:
        v = net.res_bus.vm_pu
        out.update({"vm_min": round(float(v.min()), 4), "vm_max": round(float(v.max()), 4),
                    "q_gen_mvar": round(float(rg.q_mvar.sum()), 1)})
    lam = net.res_bus.lam_p
    out["lmp_min"] = round(float(lam.min()), 2)
    out["lmp_median"] = round(float(lam.median()), 2)
    out["lmp_max"] = round(float(lam.max()), 2)
    shed = float(net.load.p_mw[~hv_l].sum() - load_p)
    out["load_shed_mw"] = round(shed, 1)
    flags = []
    if ac and not 0.5 <= out["losses_%"] <= 6:
        flags.append(f"AC losses {out['losses_%']}% outside 0.5-6%")
    if out["lines_over_100%"]:
        flags.append(f"{out['lines_over_100%']} lines >100%")
    if shed > 1:
        flags.append(f"{shed:.0f} MW load shed")
    if out["lmp_max"] > 1000 or out["lmp_min"] < -100:
        flags.append("LMP outside [-100, 1000]")
    out["flags"] = flags
    return out


def solve(cfg, net_path, out_dir):
    net = pp.from_json(net_path)
    res = {"net": str(net_path)}
    ndc, ok, dt = dc_opf(net)
    res["dc"] = {"converged": ok, "seconds": round(dt, 2)}
    if ok:
        res["dc"].update(sanity(ndc, ac=False))
        lam = ndc.res_bus.lam_p
        res["dc"]["lmp_by_kv_median"] = lam.groupby(ndc.bus.vn_kv).median().round(2).to_dict()
        top = ndc.res_line.loading_percent.sort_values(ascending=False).head(10)
        res["dc"]["top_congested"] = [(ndc.line.name[i], round(v, 1)) for i, v in top.items()]
    # warm-start AC from the DC dispatch
    if ok:
        net.gen.p_mw = ndc.res_gen.p_mw.values
    nac, rec = ac_opf(net, cfg)
    res["ac"] = rec
    if rec["feasible"]:
        res["ac"].update(sanity(nac, ac=True))
        pp.to_json(nac, out_dir / "ac_solution_net.json")
        nac.res_bus.to_csv(out_dir / "ac_res_bus.csv")
        nac.res_gen.assign(name=nac.gen.name).to_csv(out_dir / "ac_res_gen.csv")
        nac.res_line.assign(name=nac.line.name).to_csv(out_dir / "ac_res_line.csv")
    if ok:
        ndc.res_bus.to_csv(out_dir / "dc_res_bus.csv")
    (out_dir / "solve_summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res
