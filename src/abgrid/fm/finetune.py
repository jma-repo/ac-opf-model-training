"""Phase 6: zero-shot vs fine-tuned GridSFM on Alberta scenarios (CPU).

Splits (fixed, stratified by family, seed in config):
  test 150 · val 100 · train pool (rest). Each FT round starts from the
  released v1.1 weights and trains on the first n of the shuffled pool.
Metrics on the test split, against the AC-OPF solver:
  V / theta / Pg / Qg MAE (feasible cases), cost error %, feasibility
  accuracy / F1 / ECE, inference time vs solver time.
Run in .venv-fm:  python -I -m abgrid.fm.finetune --run s1250c
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from abgrid.config import load_config  # noqa: E402
from abgrid.fm.dataset import AlbertaScenarios, load_labelled, make_transform, train_dataset  # noqa: E402

CKPT = ROOT / "data/models/gridsfm/gridsfm_open_v1.1.pt"


def splits(labels: pd.DataFrame, cfg) -> dict:
    f = cfg["finetune"]
    rng = np.random.default_rng(f["seed"])
    ids = labels.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    test, val = [], []
    for fam, g in ids.groupby("family"):
        k_t = round(f["n_test"] * len(g) / len(ids)); k_v = round(f["n_val"] * len(g) / len(ids))
        test += g.id[:k_t].tolist(); val += g.id[k_t:k_t + k_v].tolist()
    pool = [i for i in ids.id if i not in set(test) | set(val)]
    return {"test": test, "val": val, "train_pool": pool}


@torch.no_grad()
def evaluate(model, paths, labels: pd.DataFrame, transform) -> dict:
    from gridsfm.data import prepare_for_inference
    model.eval()
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        g = load_labelled(p)
        g = transform(g)
        prepare_for_inference(g)
        t = time.perf_counter()
        out = model(g)
        dt = time.perf_counter() - t
        V = out["bus"].pred[:, 1].numpy(); th = out["bus"].pred[:, 0].numpy()
        Pg = out["generator"].pred[:, 0].numpy(); Qg = out["generator"].pred[:, 1].numpy()
        prob = float(torch.sigmoid(out.feas_logit.view(-1)[0]))
        sid = int(d["metadata"]["scenario_id"]); lab = labels.loc[sid]
        r = {"id": sid, "family": lab.family, "feasible": bool(lab.feasible), "p_feas": prob,
             "infer_s": dt, "solver_s": float(lab.ac_s_nominal)}
        if lab.feasible:
            s = d["solution"]["nodes"]; base = d["grid"]["context"][0][0][0]
            vt = np.array([b[1] for b in s["bus"]]); tt = np.array([b[0] for b in s["bus"]])
            pt = np.array([x[0] for x in s["generator"]]); qt = np.array([x[1] for x in s["generator"]])
            ref = [i for i, b in enumerate(d["grid"]["nodes"]["bus"]) if b[1] == 3][0]
            gen = np.array(d["grid"]["nodes"]["generator"])
            cost = lambda pg: float(np.sum(gen[:, 8] * pg ** 2 + gen[:, 9] * pg + gen[:, 10]))  # noqa: E731
            r.update(V_mae=float(np.abs(V - vt).mean()),
                     theta_mae_deg=math.degrees(float(np.abs((th - th[ref]) - (tt - tt[ref])).mean())),
                     Pg_mae_mw=float(np.abs(Pg - pt).mean() * base), Qg_mae_mvar=float(np.abs(Qg - qt).mean() * base),
                     cost_err_pct=100 * (cost(Pg) - cost(pt)) / max(abs(cost(pt)), 1e-9))
        rows.append(r)
    df = pd.DataFrame(rows)
    y, p = df.feasible.values.astype(int), df.p_feas.values
    yhat = (p >= 0.5).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    # F1 on the minority (infeasible) class, which is what screening needs
    tpi, fpi, fni = tn, fn, fp
    f1_inf = 2 * tpi / max(2 * tpi + fpi + fni, 1)
    bins = np.linspace(0, 1, 11); ece = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    fz = df[df.feasible]
    return {"n": len(df), "n_feasible": int(y.sum()),
            "V_mae_pu": round(float(fz.V_mae.mean()), 4), "theta_mae_deg": round(float(fz.theta_mae_deg.mean()), 3),
            "Pg_mae_mw": round(float(fz.Pg_mae_mw.mean()), 2), "Qg_mae_mvar": round(float(fz.Qg_mae_mvar.mean()), 2),
            "cost_err_abs_pct": round(float(fz.cost_err_pct.abs().mean()), 2),
            "cost_err_mean_pct": round(float(fz.cost_err_pct.mean()), 2),
            "feas_accuracy": round(float((yhat == y).mean()), 3), "infeasible_f1": round(f1_inf, 3),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "ece": round(float(ece), 3),
            "infer_s_median": round(float(df.infer_s.median()), 3),
            "solver_s_median": round(float(df.solver_s.median()), 2),
            "speedup_median": round(float((df.solver_s / df.infer_s).median()), 1)}, df


def main(argv=None):
    from torch_geometric.loader import DataLoader
    from gridsfm import load_model
    from gridsfm.finetune_opfdata import finetune_opfdata
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--rounds", default=None, help="comma list of n, e.g. 10,100,1000")
    a = ap.parse_args(argv)
    cfg = load_config(); f = cfg["finetune"]
    run = a.run or f["scenario_run"]
    sdir = ROOT / cfg["project"]["interim_dir"] / "scenarios" / run
    labels = pd.read_csv(sdir / "labels.csv").set_index("id", drop=False)
    P = lambda i: sdir / "pyg" / f"scn_{int(i):05d}.pyg.json"  # noqa: E731
    sp = splits(labels.reset_index(drop=True), cfg)
    out = ROOT / "outputs" / "phase6"; out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps(sp))
    torch.manual_seed(f["seed"]); torch.set_num_threads(f["threads"])
    tf = make_transform()
    test_paths = [P(i) for i in sp["test"]]
    results = {"config": f, "run": run, "splits": {k: len(v) for k, v in sp.items()}, "rounds": {}}
    res_path = out / "finetune_results.json"
    if res_path.exists():
        results["rounds"] = json.loads(res_path.read_text()).get("rounds", {})

    if "0" not in results["rounds"]:
        m = load_model(CKPT, device="cpu")
        met, df = evaluate(m, test_paths, labels, tf)
        df.to_csv(out / "test_predictions_n0.csv", index=False)
        results["rounds"]["0"] = {"n_train": 0, "test": met, "log": []}
        res_path.write_text(json.dumps(results, indent=1)); print("0-shot", met)

    rounds = [int(x) for x in (a.rounds.split(",") if a.rounds else f["rounds"])]
    val_ds = AlbertaScenarios([P(i) for i in sp["val"]], transform=tf)
    for n in rounds:
        if str(n) in results["rounds"]:
            continue
        ids = sp["train_pool"][:n]
        feas = [P(i) for i in ids if labels.loc[i, "feasible"]]
        inf = [P(i) for i in ids if not labels.loc[i, "feasible"]]
        tr = train_dataset(feas, inf, tf, infeas_prob=f["infeas_prob"], seed=f["seed"])
        epochs = f["epochs"].get(str(n), f["epochs"]["default"]) if isinstance(f["epochs"], dict) else f["epochs"]
        m = load_model(CKPT, device="cpu")
        t0 = time.time()
        log = finetune_opfdata(m, DataLoader(tr, batch_size=f["batch_size"], shuffle=True),
                               val_loader=DataLoader(val_ds, batch_size=f["batch_size"]),
                               epochs=epochs, lr=f["lr"],
                               on_epoch_end=lambda e: print(f"n={n} epoch {e['epoch']} train {e['train_loss']:.4f} "
                                                            f"val {e.get('val_loss', float('nan')):.4f} {e['epoch_s']}s", flush=True))
        ck = ROOT / "data/models/ft" / f"gridsfm_alberta_n{n}.pt"; ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "n_train": n, "epochs": epochs}, ck)
        met, df = evaluate(m, test_paths, labels, tf)
        df.to_csv(out / f"test_predictions_n{n}.csv", index=False)
        results["rounds"][str(n)] = {"n_train": n, "n_train_feasible": len(feas), "n_train_infeasible": len(inf),
                                     "epochs": epochs, "train_s": round(time.time() - t0, 1), "test": met,
                                     "log": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                              for k, v in e.items() if not isinstance(v, (dict, list))} for e in log]}
        res_path.write_text(json.dumps(results, indent=1, default=float)); print(f"FT-{n}", met, flush=True)
    return results


if __name__ == "__main__":
    main()
