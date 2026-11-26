"""Cost calibration by inverse optimisation against AESO metered dispatch.

Structural fixes (always applied by scenarios.apply_spec):
  * load = AESO planning-area grid-served `Load` (excl. behind-the-fence)
  * cogeneration, hydro, biomass pinned to metered output each hour
  * wind/solar availability = metered output
  * interchange = metered importer - exporter, split by tie rating
Fitted: for each price-dispatched gas fuel f, unit cost
    c1_u = a_f + b_f * (1 - CF_u)
where CF_u is the unit's 2024 metered capacity factor. (a, b) are fitted with
Nelder-Mead so DC-OPF dispatch matches metered dispatch on fit hours, then
checked with AC-OPF on held-out hours. Result: data/interim/params/calibration.json
(provenance derived:inverse_opt_metered_dispatch_2024)."""
from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import pandapower as pp
from scipy.optimize import minimize

from ..config import path
from ..scenarios import generate as sg
from ..solve.opf import ac_opf, dc_opf

log = logging.getLogger("abgrid.calibrate")
_W: dict = {}


def _net_import(cfg) -> pd.Series:
    raw = path(cfg, "raw_dir")
    out = []
    for f in sorted((raw / "aeso").glob("metered_volume_*.csv.gz")):
        d = pd.read_csv(f)
        x = d[d.asset_class.isin(["IMPORTER", "EXPORTER"])].groupby(["begin_date_utc", "asset_class"]) \
            .metered_volume_mwh.sum().unstack().fillna(0)
        out.append(x.IMPORTER - x.EXPORTER)
    s = pd.concat(out)
    s.index = pd.to_datetime(s.index).tz_localize("UTC").tz_convert("America/Edmonton").tz_localize(None)
    return s


def _hours(cfg, inp):
    c = cfg["calibration"]
    ni = _net_import(cfg)
    ok = [h for h in inp["hours"] if h in ni.index and any(pd.Timestamp(a) <= h <= pd.Timestamp(b) + pd.Timedelta(hours=23)
                                                           for a, b in c["hour_windows"])]
    rng = np.random.default_rng(c["seed"])
    pick = sorted(rng.choice(len(ok), c["n_fit_hours"] + c["n_val_hours"], replace=False))
    hrs = [ok[i] for i in pick]
    rng.shuffle(hrs)
    return hrs[: c["n_fit_hours"]], hrs[c["n_fit_hours"]:], ni


def _capacity_factor(inp, units) -> pd.Series:
    um = inp["umap"].loc[units]
    a = inp["mv"].reindex(columns=um.asset.unique())
    cf = {u: float((a[um.asset[u]] * um.share[u]).mean() / max(um.p_max_mw[u], 1))
          if um.asset[u] in a.columns and a[um.asset[u]].notna().any() else np.nan for u in units}
    return pd.Series(cf)


def costs_from_params(theta, fuels, units_by_fuel, cf, lo, hi) -> dict:
    c1 = {}
    for k, f in enumerate(fuels):
        a = float(np.clip(theta[2 * k], *lo)); b = float(np.clip(theta[2 * k + 1], *hi))
        for u in units_by_fuel[f]:
            c1[u] = a + b * (1 - cf[u])
    return c1


def _spec(h, ni, cfg, i):
    return {"id": i, "family": "calib", "hour": str(h), "noise": 0.0, "noise_seed": 0,
            "hvdc_mw": dict(cfg["topology"]["hvdc_schedule_mw"]), "intertie_u": {},
            "net_import_mw": float(ni.get(h, 0.0))}


def _init(cfg):
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    inp = sg._inputs(cfg)
    inp["cal"] = None
    _W.update(cfg=cfg, inp=inp, base=pp.from_json(inp["net"]))


def _eval_hour(args):
    sp, c1, c2, ac = args
    cfg, inp = _W["cfg"], _W["inp"]
    inp = {**inp, "cal": {"c1_by_unit": c1, "c2": c2, "use_costs": True} if c1 is not None else None}
    n = sg.apply_spec(cfg, _W["base"], inp, sp)
    if ac:
        sol, rec = ac_opf(n, {**cfg, "opf": {**cfg["opf"], "relaxations": []}})
        ok = rec["feasible"]
    else:
        sol, ok, _ = dc_opf(n)
    if not ok:
        return sp["hour"], None
    return sp["hour"], dict(zip(sol.gen.name, sol.res_gen.p_mw.values))


def _actual(inp, hours, units):
    return pd.DataFrame({h: sg.unit_actual(inp, pd.Timestamp(h)).reindex(units) for h in hours}).T


def _score(disp: dict, act: pd.DataFrame, units, fuel_of) -> tuple[float, dict]:
    err, fuel_err, n = 0.0, 0.0, 0
    for h, d in disp.items():
        if d is None:
            err += 1e7; continue
        a = act.loc[h]
        p = pd.Series({u: d.get(u, 0.0) for u in units})
        m = a.notna()
        err += float(((p[m] - a[m]) ** 2).sum())
        ft = (p[m] - a[m]).groupby(fuel_of[m.index[m]]).sum()
        fuel_err += float((ft ** 2).sum())
        n += int(m.sum())
    return (err / max(n, 1)) ** 0.5 + 0.2 * (fuel_err / max(len(disp), 1)) ** 0.5, {}


def metrics(disp: dict, act: pd.DataFrame, units, fuel_of) -> dict:
    P, A = [], []
    fuel_rows = []
    for h, d in disp.items():
        if d is None:
            continue
        a = act.loc[h]; m = a.notna()
        p = pd.Series({u: d.get(u, 0.0) for u in units})
        P += list(p[m]); A += list(a[m])
        fuel_rows.append(pd.DataFrame({"opf": p[m], "actual": a[m], "fuel": fuel_of[m.index[m]].values})
                         .groupby("fuel").sum())
    P, A = np.array(P), np.array(A)
    fz = pd.concat(fuel_rows).groupby(level=0).mean().round(0) if fuel_rows else pd.DataFrame()
    return {"unit_mae_mw": round(float(np.abs(P - A).mean()), 1),
            "unit_corr": round(float(np.corrcoef(P, A)[0, 1]), 3),
            "n_unit_hours": len(P), "hours_solved": sum(d is not None for d in disp.values()),
            "fuel_mean_mw": fz.to_dict() if len(fz) else {}}


def run(cfg):
    t0 = time.perf_counter()
    c = cfg["calibration"]
    inp = sg._inputs(cfg)
    fuels = list(c["dispatched_fuels"])
    G = inp["gens"].set_index("unit_code")
    units_by_fuel = {f: list(G.index[G.fuel == f]) for f in fuels}
    disp_units = [u for f in fuels for u in units_by_fuel[f]]
    cf = _capacity_factor(inp, disp_units)
    for f in fuels:   # unmetered units: fuel median capacity factor
        cf[units_by_fuel[f]] = cf[units_by_fuel[f]].fillna(cf[units_by_fuel[f]].median())
    fit_h, val_h, ni = _hours(cfg, inp)
    specs_fit = [_spec(h, ni, cfg, i) for i, h in enumerate(fit_h)]
    specs_val = [_spec(h, ni, cfg, 1000 + i) for i, h in enumerate(val_h)]
    # score dispatched gas units plus fixed-output units are pinned anyway
    fuel_of = G.fuel
    act_fit = _actual(inp, [s["hour"] for s in specs_fit], disp_units)
    act_val = _actual(inp, [s["hour"] for s in specs_val], disp_units)
    lo, hi = c["bounds"]
    theta0 = np.array([v for f in fuels for v in c["init"][f]], float)
    illustrative = {u: cfg["params"]["fuels"][fuel_of[u]]["c1"] for u in disp_units}
    hist = []
    with mp.get_context("fork").Pool(cfg["scenarios"]["n_processes"], initializer=_init, initargs=(cfg,)) as pool:
        def solve_all(specs, c1, ac=False):
            return dict(pool.map(_eval_hour, [(s, c1, c["c2"], ac) for s in specs]))

        def obj(theta):
            c1 = costs_from_params(theta, fuels, units_by_fuel, cf, lo, hi)
            s, _ = _score(solve_all(specs_fit, c1), act_fit, disp_units, fuel_of)
            hist.append({"theta": [round(x, 2) for x in theta], "score": round(s, 3)})
            log.info("eval %d: theta=%s score=%.3f", len(hist), np.round(theta, 1).tolist(), s)
            return s

        before_fit = _score(solve_all(specs_fit, illustrative), act_fit, disp_units, fuel_of)[0]
        # wide initial simplex: +step on each parameter (CAD/MWh)
        simplex = np.vstack([theta0] + [theta0 + c["simplex_step"] * np.eye(len(theta0))[k]
                                        for k in range(len(theta0))])
        res = minimize(obj, theta0, method="Nelder-Mead",
                       options={"maxfev": c["max_evals"], "xatol": 0.5, "fatol": 0.05, "initial_simplex": simplex})
        best = res.x
        c1_best = costs_from_params(best, fuels, units_by_fuel, cf, lo, hi)
        # held-out validation with AC-OPF: illustrative vs calibrated (both with structural fixes)
        val_before = metrics(solve_all(specs_val, illustrative, ac=True), act_val, disp_units, fuel_of)
        val_after = metrics(solve_all(specs_val, c1_best, ac=True), act_val, disp_units, fuel_of)
    out = {
        "fuels": fuels,
        "params": {f: {"a": round(float(np.clip(best[2 * k], *lo)), 2), "b": round(float(np.clip(best[2 * k + 1], *hi)), 2)}
                   for k, f in enumerate(fuels)},
        "c2": c["c2"], "c1_by_unit": {u: round(v, 3) for u, v in c1_best.items()},
        "capacity_factor_2024": {u: round(float(cf[u]), 3) for u in disp_units},
        "fit_hours": [s["hour"] for s in specs_fit], "val_hours": [s["hour"] for s in specs_val],
        "score_fit_illustrative": round(before_fit, 3), "score_fit_calibrated": round(float(res.fun), 3),
        "n_evals": len(hist), "validation_ac": {"illustrative_costs": val_before, "calibrated_costs": val_after},
        "history": hist, "seconds": round(time.perf_counter() - t0, 1),
        "provenance": "derived:inverse_opt_metered_dispatch_2024",
    }
    p = path(cfg, "interim_dir") / "params" / "calibration.json"
    p.write_text(json.dumps(out, indent=1))
    log.info("calibration done: %s", {k: out[k] for k in ("params", "score_fit_illustrative", "score_fit_calibrated")})
    return out
