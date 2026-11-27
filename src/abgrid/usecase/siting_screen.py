"""Phase 7: data-centre siting screen with the fine-tuned GridSFM, verified by AC-OPF.

  python -m abgrid.usecase.siting_screen base      (.venv)    solve + export base snapshots
  python -I src/abgrid/usecase/siting_screen.py screen  (.venv-fm)  model on every candidate
  python -m abgrid.usecase.siting_screen verify    (.venv)    AC-OPF on selected candidates
  python -m abgrid.usecase.siting_screen report    (.venv)    tables, map, report

Every candidate = base snapshot + one new load of size S at a 138/240 kV bus
(pf 0.98). The model predicts P(feasible at nominal limits) and dispatch; cost
impact = predicted cost(candidate) - predicted cost(base), both from the model
so that its bias largely cancels. The solver is then run on top, borderline,
worst and random candidates. Illustrative costs; not AESO's grid.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
WORK = ROOT / "data/interim/siting"
OUT = ROOT / "outputs/phase7"


def _snapshot_hours(cfg, inp):
    import pandas as pd
    hrs = {}
    for tag, h in cfg["siting"]["snapshots"].items():
        if h == "auto_max_load_jul":
            win = [w for w in cfg["sources"]["aeso"]["metered_volume_windows"] if "-07-" in str(w[0])][0]
            pa = inp["pa"]; tot = pa.sum(axis=1)
            m = tot[(tot.index >= pd.Timestamp(win[0])) & (tot.index <= pd.Timestamp(win[1]) + pd.Timedelta(hours=23))]
            hrs[tag] = m.idxmax()
        else:
            hrs[tag] = pd.Timestamp(h)
    return hrs


def _spec(cfg, h, ni, sid):
    return {"id": sid, "family": "siting", "hour": str(h), "noise": 0.0, "noise_seed": 0,
            "hvdc_mw": dict(cfg["topology"]["hvdc_schedule_mw"]), "intertie_u": {},
            "net_import_mw": float(ni.get(h, 0.0))}


def base(cfg=None):
    import logging
    import pandapower as pp
    from abgrid.config import load_config
    from abgrid.export import case as ex
    from abgrid.params.calibrate import _net_import
    from abgrid.scenarios import generate as sg
    from abgrid.solve.opf import ac_opf, sanity
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    cfg = cfg or load_config(); S = cfg["siting"]
    inp = sg._inputs(cfg); net0 = pp.from_json(inp["net"]); ni = _net_import(cfg)
    WORK.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    buses = _bus_table(cfg)
    info = {}
    for k, (tag, h) in enumerate(_snapshot_hours(cfg, inp).items()):
        sp = _spec(cfg, h, ni, 90000 + k)
        n = sg.apply_spec(cfg, net0, inp, sp)
        sol, rec = ac_opf(n, {**cfg, "opf": {**cfg["opf"], "relaxations": []}})
        assert rec["feasible"], f"base {tag} not feasible at nominal limits"
        sc = sanity(sol, ac=True)
        d = ex.to_gridsfm(ex.build_case(sol, solved=True), {"scenario_id": sp["id"], "snapshot": tag, "hour_mpt": str(h)})
        (WORK / f"base_{tag}.pyg.json").write_text(json.dumps(d, default=ex._json_default))
        bi = list(n.bus.index)
        key = [(n.bus.name[b], int(n.bus.vn_kv[b])) for b in bi]
        bnd = [bool(buses.is_boundary.get(k, False)) for k in key]
        cand = [p for p, b in enumerate(bi) if int(n.bus.vn_kv[b]) in S["candidate_kv"] and not bnd[p]]
        info[tag] = {"hour": str(h), "spec": sp, "cost": sc["cost"], "load_mw": sc.get("load_mw"),
                     "lmp_median": sc["lmp_median"], "ac_s": rec["attempts"][0]["seconds"],
                     "bus_ids": bi, "bus_keys": key, "candidates": cand}
        print(tag, h, "cost", round(sc["cost"]), "candidates", len(cand), flush=True)
    (WORK / "base_info.json").write_text(json.dumps(info, default=str, indent=1))


def _bus_table(cfg):
    """Topology bus table keyed by (node_code, kV), which is how the
    pandapower net names its buses."""
    import pandas as pd
    b = pd.read_csv(path_topology(cfg) / "buses.csv")
    return b.set_index(["node_code", "base_kv"])


def path_topology(cfg):
    return ROOT / cfg["project"]["interim_dir"] / "topology"


def _with_load(d, pos, mw, pf, base_mva):
    g = copy.deepcopy(d)
    g.pop("solution", None)
    g["grid"]["nodes"]["load"].append([mw / base_mva, mw * math.tan(math.acos(pf)) / base_mva])
    ll = g["grid"]["edges"]["load_link"]
    ll["senders"].append(len(g["grid"]["nodes"]["load"]) - 1); ll["receivers"].append(pos)
    g["metadata"]["feasible"] = False
    return g


def screen(cfg=None):
    import pandas as pd
    import torch
    from gridsfm import load_model
    from gridsfm.data import load_opfdata, prepare_for_inference
    from abgrid.config import load_config
    from abgrid.fm.dataset import make_transform
    from abgrid.fm.finetune import CKPT
    cfg = cfg or load_config(); S = cfg["siting"]
    torch.set_num_threads(cfg["finetune"]["threads"])
    m = load_model(CKPT, device="cpu")
    m.load_state_dict(torch.load(ROOT / "data/models/ft" / S["checkpoint"])["state_dict"]); m.eval()
    tf = make_transform()
    info = json.loads((WORK / "base_info.json").read_text())
    tmp = WORK / "_cand.pyg.json"

    def run(d):
        tmp.write_text(json.dumps(d))
        g = tf(load_opfdata(tmp)); prepare_for_inference(g)
        with torch.no_grad():
            o = m(g)
        gen = np.array(d["grid"]["nodes"]["generator"]); pg = o["generator"].pred[:, 0].numpy()
        cost = float(np.sum(gen[:, 8] * pg ** 2 + gen[:, 9] * pg + gen[:, 10]))
        return float(torch.sigmoid(o.feas_logit.view(-1)[0])), cost, o["bus"].pred[:, 1].numpy()

    rows = []
    for tag, inf in info.items():
        d = json.loads((WORK / f"base_{tag}.pyg.json").read_text())
        bm = d["grid"]["context"][0][0][0]
        p0, c0, _ = run({k: v for k, v in d.items() if k != "solution"})
        t0 = time.time()
        for pos in inf["candidates"]:
            for mw in S["sizes_mw"]:
                t = time.perf_counter()
                p, c, vm = run(_with_load(d, pos, mw, S["pf"], bm))
                rows.append({"snapshot": tag, "bus_pos": pos, "bus_id": inf["bus_ids"][pos], "size_mw": mw,
                             "p_feasible": p, "pred_cost": c, "pred_base_cost": c0, "pred_dcost": c - c0,
                             "pred_dcost_per_mwh": (c - c0) / mw, "pred_vm_min": float(vm.min()),
                             "infer_s": time.perf_counter() - t})
            if len(rows) % 300 < len(S["sizes_mw"]):
                print(tag, len(rows), f"{time.time() - t0:.0f}s", flush=True)
        inf["pred_base"] = {"p_feasible": p0, "cost": c0}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "screen_predictions.csv", index=False)
    (WORK / "base_info.json").write_text(json.dumps(info, default=str, indent=1))
    tmp.unlink(missing_ok=True)
    print("screened", len(rows))


def _select(df, S):
    v = S["verify"]; rng = np.random.default_rng(v["seed"]); pick = []
    for (tag, mw), g in df.groupby(["snapshot", "size_mw"]):
        feas = g[g.p_feasible >= 0.5].sort_values("pred_dcost")
        sel = {"top": feas.head(v["top"]),
               "borderline": g.iloc[(g.p_feasible - 0.5).abs().argsort()[: v["borderline"]]],
               "worst": g.sort_values("p_feasible").head(v["worst"])}
        used = set().union(*[set(x.index) for x in sel.values()])
        rest = g.drop(index=list(used))
        sel["random"] = rest.iloc[rng.choice(len(rest), min(v["random"], len(rest)), replace=False)]
        for why, x in sel.items():
            for i in x.index:
                pick.append((i, why))
    seen, out = set(), []
    for i, why in pick:
        if i not in seen:
            seen.add(i); out.append((i, why))
    return out


_W: dict = {}


def _init(cfg):
    import logging
    import pandapower as pp
    from abgrid.scenarios import generate as sg
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    inp = sg._inputs(cfg)
    _W.update(cfg=cfg, inp=inp, base=pp.from_json(inp["net"]))


def _verify_one(args):
    from abgrid.scenarios import generate as sg
    from abgrid.solve.opf import ac_opf, dc_opf, sanity
    key, sp = args
    cfg = _W["cfg"]
    n = sg.apply_spec(cfg, _W["base"], _W["inp"], sp)
    t = time.perf_counter()
    ndc, ok, _ = dc_opf(n)
    if ok:
        n.gen.p_mw = ndc.res_gen.p_mw.values
    sol, rec = ac_opf(n, cfg)
    r = {"key": key, "feasible_any": rec["feasible"], "relaxation_level": rec["relaxation_level"],
         "feasible": bool(rec["feasible"] and rec["relaxation_level"] == 0),
         "ac_s_nominal": rec["attempts"][0]["seconds"], "solver_s": round(time.perf_counter() - t, 2)}
    if rec["feasible"]:
        sc = sanity(sol, ac=True)
        r.update(cost=sc["cost"], load_shed_mw=sc["load_shed_mw"], vm_min=sc["vm_min"], lmp_median=sc["lmp_median"])
    return r


def verify(cfg=None):
    import multiprocessing as mp
    import pandas as pd
    from abgrid.config import load_config
    cfg = cfg or load_config(); S = cfg["siting"]
    df = pd.read_csv(OUT / "screen_predictions.csv")
    info = json.loads((WORK / "base_info.json").read_text())
    sel = _select(df, S)
    jobs = []
    for i, why in sel:
        r = df.loc[i]; sp = dict(info[r.snapshot]["spec"])
        sp.update(id=int(i), dc_bus=int(r.bus_id), dc_mw=float(r.size_mw))
        jobs.append((int(i), sp))
    t0 = time.time()
    with mp.get_context("fork").Pool(cfg["scenarios"]["n_processes"], initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_verify_one, jobs, chunksize=1)
    v = pd.DataFrame(res).set_index("key")
    out = df.loc[[i for i, _ in sel]].copy()
    out["why"] = [w for _, w in sel]
    out = out.join(v)
    out["base_cost"] = out.snapshot.map({k: x["cost"] for k, x in info.items()})
    out["true_dcost"] = out.cost - out.base_cost
    out.to_csv(OUT / "verify_results.csv", index_label="row")
    print("verified", len(out), f"{time.time() - t0:.0f}s")


def report(cfg=None):
    import pandas as pd
    from abgrid.config import load_config
    cfg = cfg or load_config()
    df = pd.read_csv(OUT / "screen_predictions.csv")
    v = pd.read_csv(OUT / "verify_results.csv")
    info = json.loads((WORK / "base_info.json").read_text())
    buses = _bus_table(cfg)
    geo = {}
    for tag, x in info.items():
        for bid, k in zip(x["bus_ids"], x["bus_keys"]):
            r = buses.loc[tuple(k)]
            geo[bid] = {"lon": r.lon, "lat": r.lat, "name": r["name"], "base_kv": k[1]}
    geo = pd.DataFrame.from_dict(geo, orient="index")
    y, p = v.feasible.astype(int).values, v.p_feasible.values
    yhat = (p >= 0.5).astype(int)
    conf = {"tp": int(((yhat == 1) & (y == 1)).sum()), "fp": int(((yhat == 1) & (y == 0)).sum()),
            "fn": int(((yhat == 0) & (y == 1)).sum()), "tn": int(((yhat == 0) & (y == 0)).sum())}
    fz = v[v.feasible & v.cost.notna()]
    by_why = v.groupby("why").apply(lambda g: pd.Series({
        "n": len(g), "solver_feasible": int(g.feasible.sum()), "model_says_feasible": int((g.p_feasible >= 0.5).sum()),
        "agree": int(((g.p_feasible >= 0.5) == g.feasible).sum())}), include_groups=False)
    summ = {"snapshots": {k: {kk: x[kk] for kk in ("hour", "cost", "lmp_median", "ac_s")} | {"n_candidates": len(x["candidates"])}
                          for k, x in info.items()},
            "n_screened": len(df), "screen_infer_s_total": round(float(df.infer_s.sum()), 1),
            "screen_infer_s_median": round(float(df.infer_s.median()), 3),
            "solver_s_median_verified": round(float(v.solver_s.median()), 2),
            "brute_force_solver_hours_est": round(float(len(df) * v.solver_s.median() / 3600), 2),
            "predicted_feasible_share": df.groupby(["snapshot", "size_mw"]).p_feasible.apply(lambda s: round(float((s >= 0.5).mean()), 3)).unstack().to_dict(),
            "verify_n": len(v), "verify_confusion": conf,
            "verify_accuracy": round(float((yhat == y).mean()), 3),
            "verify_by_reason": by_why.to_dict(orient="index"),
            "dcost_mae_cad_h": round(float((fz.pred_dcost - fz.true_dcost).abs().mean()), 0),
            "dcost_corr": round(float(np.corrcoef(fz.pred_dcost, fz.true_dcost)[0, 1]), 3) if len(fz) > 2 else None,
            "true_dcost_per_mwh_median": round(float((fz.true_dcost / fz.size_mw).median()), 1)}
    (OUT / "siting_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    # map: p_feasible at 300 MW per snapshot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    br = json.loads((path_topology(cfg) / "branches.geojson").read_text())
    pos2 = df.drop(columns=[c for c in ("lon", "lat", "name", "base_kv") if c in df], errors="ignore") \
        .merge(geo, left_on="bus_id", right_index=True, how="left")
    pos2.to_csv(OUT / "screen_predictions.csv", index=False)
    fig, axes = plt.subplots(1, len(info), figsize=(7 * len(info), 8), dpi=150)
    for ax, tag in zip(np.atleast_1d(axes), info):
        for f in br["features"]:
            c = f["geometry"]["coordinates"]
            if f["geometry"]["type"] == "LineString":
                xs, ys = zip(*c); ax.plot(xs, ys, color="#c9ced6", lw=0.5, zorder=1)
        s = pos2[(pos2.snapshot == tag) & (pos2.size_mw == 300)]
        sc = ax.scatter(s.lon, s.lat, c=s.p_feasible, cmap="RdYlGn", vmin=0, vmax=1, s=14, edgecolor="k", lw=0.2, zorder=3)
        vv = v[(v.snapshot == tag) & (v.size_mw == 300)].merge(geo[["lon", "lat"]], left_on="bus_id", right_index=True)
        ax.scatter(vv.lon, vv.lat, s=70, facecolor="none", edgecolor=np.where(vv.feasible, "#1d7a46", "#b4232a"), lw=1.4, zorder=4)
        ax.set_title(f"{tag} {info[tag]['hour']} - 300 MW data centre\nGridSFM P(feasible); rings = solver check (green ok / red infeasible)", fontsize=9)
        ax.set_aspect(1 / math.cos(math.radians(54))); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, shrink=0.6, label="P(feasible at nominal limits)")
    fig.text(0.01, 0.01, "Approximate open-data model, NOT AESO's grid. Illustrative costs.", fontsize=8)
    fig.savefig(OUT / "siting_map_300mw.png", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    {"base": base, "screen": screen, "verify": verify, "report": report}[sys.argv[1]]()
