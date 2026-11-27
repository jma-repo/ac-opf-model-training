"""Phase 6c: does a GridSFM prediction speed up the AC-OPF solver?

Step 1 (.venv-fm):  python -I src/abgrid/fm/warmstart.py predict
    -> data/interim/warmstart/pred.npz  (V, theta, Pg for feasible test scenarios)
Step 2 (.venv):     python -m abgrid.fm.warmstart solve
    -> outputs/phase6/warmstart_results.{csv,json}

For each scenario the same pandapower net is rebuilt from its spec and solved
with PIPS from four starting points:
  flat      V = 1, theta = 0, Pg = setpoint
  dc_pf     pipeline default: DC-OPF Pg, then a power flow (init="pf")
  fm_pf     GridSFM Pg, then a power flow (init="pf")
  fm_pfv    GridSFM Pg and GridSFM V as generator voltage setpoints, then a power flow
  fm_full   GridSFM Pg and GridSFM V/theta as the initial voltage vector (init="results")
Model values are pulled 2% (Pg) / 0.005 pu (V) inside their limits, since
interior-point solvers stall when started on a bound. Only nominal limits. The DC-OPF time is charged to dc_pf
and the model inference time to fm_*.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/interim/warmstart"


def predict(n_max=60, ckpt_name=None):
    import pandas as pd
    import torch
    from gridsfm import load_model
    from gridsfm.data import prepare_for_inference
    from abgrid.config import load_config
    from abgrid.fm.dataset import load_labelled, make_transform
    from abgrid.fm.finetune import CKPT
    cfg = load_config(); f = cfg["finetune"]
    sdir = ROOT / cfg["project"]["interim_dir"] / "scenarios" / f["scenario_run"]
    labels = pd.read_csv(sdir / "labels.csv").set_index("id", drop=False)
    sp = json.loads((ROOT / "outputs/phase6/splits.json").read_text())
    ids = [i for i in sp["test"] if labels.loc[i, "feasible"]][:int(n_max)]
    torch.set_num_threads(f["threads"])
    m = load_model(CKPT, device="cpu")
    ck = ROOT / "data/models/ft" / (ckpt_name or "gridsfm_alberta_n1000_refined.pt")
    m.load_state_dict(torch.load(ck)["state_dict"]); m.eval()
    tf = make_transform(); res = {}
    for i in ids:
        g = tf(load_labelled(sdir / "pyg" / f"scn_{i:05d}.pyg.json")); prepare_for_inference(g)
        t = time.perf_counter()
        with torch.no_grad():
            o = m(g)
        dt = time.perf_counter() - t
        res[i] = dict(theta=o["bus"].pred[:, 0].numpy(), vm=o["bus"].pred[:, 1].numpy(),
                      pg=o["generator"].pred[:, 0].numpy(), qg=o["generator"].pred[:, 1].numpy(), infer_s=dt)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "pred.npy", {"ckpt": ck.name, "pred": res}, allow_pickle=True)
    print("predicted", len(res), "from", ck.name)


def _capture_iterations():
    import pandapower.optimal_powerflow as po
    box = {"it": None}
    orig = po.opf

    def wrapped(ppci, ppopt):
        r = orig(ppci, ppopt)
        try:
            box["it"] = int(r["raw"]["output"]["iterations"])
        except Exception:  # noqa: BLE001
            box["it"] = None
        return r
    po.opf = wrapped
    return box


def _interior_pg(n0, gi, pg_pu, margin=0.02):
    """Predicted Pg (MW) pulled strictly inside [Pmin, Pmax]: interior-point
    solvers stall when started on a bound."""
    lo, hi = n0.gen.min_p_mw[gi].values, n0.gen.max_p_mw[gi].values
    span = np.maximum(hi - lo, 0.0)
    return np.clip(pg_pu * 100.0, lo + margin * span, hi - margin * span)


def solve():
    import copy
    import logging
    import pandas as pd
    import pandapower as pp
    from abgrid.config import load_config
    from abgrid.scenarios import generate as sg
    from abgrid.solve.opf import dc_opf
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    cfg = load_config(); f = cfg["finetune"]
    sdir = ROOT / cfg["project"]["interim_dir"] / "scenarios" / f["scenario_run"]
    specs = {s["id"]: s for s in map(json.loads, (sdir / "specs.jsonl").read_text().splitlines())}
    labels = pd.read_csv(sdir / "labels.csv").set_index("id", drop=False)
    P = np.load(OUT / "pred.npy", allow_pickle=True).item()
    inp = sg._inputs(cfg); base = pp.from_json(inp["net"])
    box = _capture_iterations()
    rows = []

    def run(n, init):
        t = time.perf_counter()
        try:
            pp.runopp(n, init=init, verbose=False, calculate_voltage_angles=True, numba=False, OPF_FLOW_LIM="S")
            ok = bool(n.OPF_converged)
        except Exception:  # noqa: BLE001
            ok = False
        return ok, time.perf_counter() - t, (float(n.res_cost) if ok else None), box["it"]

    for i, pr in P["pred"].items():
        n0 = sg.apply_spec(cfg, base, inp, specs[i])
        gi = n0.gen.index[n0.gen.in_service]          # same order as the exported generator rows
        bi = n0.bus.index
        r = {"id": i, "family": labels.loc[i, "family"], "label_cost": labels.loc[i, "cost"], "infer_s": pr["infer_s"]}
        # flat
        n = copy.deepcopy(n0); ok, dt, c, it = run(n, "flat")
        r.update(flat_ok=ok, flat_s=dt, flat_cost=c, flat_it=it)
        # pipeline default
        ndc, dok, dct = dc_opf(n0); n = copy.deepcopy(n0)
        if dok:
            n.gen.p_mw = ndc.res_gen.p_mw.values
        ok, dt, c, it = run(n, "pf")
        r.update(dc_pf_ok=ok, dc_pf_s=dt + dct, dc_pf_cost=c, dc_pf_it=it, dc_s=dct)
        # GridSFM Pg + power flow
        n = copy.deepcopy(n0); n.gen.loc[gi, "p_mw"] = _interior_pg(n0, gi, pr["pg"])
        ok, dt, c, it = run(n, "pf")
        r.update(fm_pf_ok=ok, fm_pf_s=dt + pr["infer_s"], fm_pf_cost=c, fm_pf_it=it)
        # GridSFM Pg + GridSFM V as generator voltage setpoints, then a power flow
        n = copy.deepcopy(n0); n.gen.loc[gi, "p_mw"] = _interior_pg(n0, gi, pr["pg"])
        pos = {b: k for k, b in enumerate(bi)}
        n.gen.loc[gi, "vm_pu"] = np.clip(pr["vm"][[pos[b] for b in n0.gen.bus[gi]]], 0.955, 1.045)
        ok, dt, c, it = run(n, "pf")
        r.update(fm_pfv_ok=ok, fm_pfv_s=dt + pr["infer_s"], fm_pfv_cost=c, fm_pfv_it=it)
        # GridSFM Pg + V/theta as the starting voltage vector
        n = copy.deepcopy(n0); n.gen.loc[gi, "p_mw"] = _interior_pg(n0, gi, pr["pg"])
        th = np.degrees(pr["theta"] - pr["theta"][np.argmax(n0.bus.index.isin(n0.gen.bus[n0.gen.slack.astype(bool)]))])
        pp.runpp(n, numba=False)            # creates result tables of the right shape
        n.res_bus.loc[bi, "vm_pu"] = np.clip(pr["vm"], n0.bus.min_vm_pu.values + 0.005, n0.bus.max_vm_pu.values - 0.005); n.res_bus.loc[bi, "va_degree"] = th
        ok, dt, c, it = run(n, "results")
        r.update(fm_full_ok=ok, fm_full_s=dt + pr["infer_s"], fm_full_cost=c, fm_full_it=it)
        rows.append(r)
        print(i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k.endswith(("_s", "_it", "_ok"))}, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/phase6"; df.to_csv(out / "warmstart_results.csv", index=False)
    summ = {"n": len(df), "checkpoint": P["ckpt"]}
    for k in ("flat", "dc_pf", "fm_pf", "fm_pfv", "fm_full"):
        okm = df[f"{k}_ok"]
        summ[k] = {"converged": int(okm.sum()), "median_s": round(float(df.loc[okm, f"{k}_s"].median()), 2),
                   "median_iterations": float(df.loc[okm, f"{k}_it"].median()) if df[f"{k}_it"].notna().any() else None,
                   "cost_vs_label_pct_max": round(float((100 * (df.loc[okm, f"{k}_cost"] / df.loc[okm, "label_cost"] - 1)).abs().max()), 3)}
    both = df.dc_pf_ok & df.fm_full_ok
    summ["speedup_fm_full_vs_dc_pf_median"] = (round(float((df.dc_pf_s[both] / df.fm_full_s[both]).median()), 2)
                                               if both.any() else None)
    both = df.dc_pf_ok & df.fm_pfv_ok
    summ["speedup_fm_pfv_vs_dc_pf_median"] = round(float((df.dc_pf_s[both] / df.fm_pfv_s[both]).median()), 2)
    both = df.dc_pf_ok & df.fm_pf_ok
    summ["speedup_fm_pf_vs_dc_pf_median"] = round(float((df.dc_pf_s[both] / df.fm_pf_s[both]).median()), 2)
    (out / "warmstart_results.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"predict": predict, "solve": solve}[sys.argv[1]](*sys.argv[2:])
