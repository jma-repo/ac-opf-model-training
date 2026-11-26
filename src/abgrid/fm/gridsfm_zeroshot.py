"""Zero-shot GridSFM inference on .pyg.json scenarios that carry a solver
`solution` block; reports error vs the AC-OPF solver.

Run inside the separate FM venv (.venv-fm, CPU torch):
    python -m abgrid.fm.gridsfm_zeroshot <scenario.pyg.json> [...]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

CKPT = "gridsfm_open_v1.1.pt"


def _cost(gen_rows, pg_pu):
    g = np.asarray(gen_rows)
    return float(np.sum(g[:, 8] * pg_pu ** 2 + g[:, 9] * pg_pu + g[:, 10]))


def evaluate(model, path: Path) -> dict:
    from gridsfm import predict
    d = json.loads(path.read_text())
    sol = d["solution"]["nodes"]
    t = time.perf_counter()
    out = predict(model, path)
    dt = time.perf_counter() - t
    va_true = np.array([b[0] for b in sol["bus"]])
    vm_true = np.array([b[1] for b in sol["bus"]])
    pg_true = np.array([g[0] for g in sol["generator"]])
    qg_true = np.array([g[1] for g in sol["generator"]])
    V, th = out["V"].numpy(), out["theta"].numpy()
    Pg, Qg = out["Pg"].numpy(), out["Qg"].numpy()
    ref = [i for i, b in enumerate(d["grid"]["nodes"]["bus"]) if b[1] == 3][0]
    th_err = (th - th[ref]) - (va_true - va_true[ref])
    base = d["grid"]["context"][0][0][0]
    gens = d["grid"]["nodes"]["generator"]
    c_true = _cost(gens, pg_true)
    c_pred = _cost(gens, Pg)
    pd_tot = sum(x[0] for x in d["grid"]["nodes"]["load"])
    return {
        "case": path.name, "n_bus": len(vm_true), "n_gen": len(pg_true),
        "infer_s": round(dt, 3),
        "V_mae_pu": round(float(np.abs(V - vm_true).mean()), 4),
        "theta_mae_deg": round(math.degrees(float(np.abs(th_err).mean())), 3),
        "Pg_mae_MW": round(float(np.abs(Pg - pg_true).mean()) * base, 2),
        "Qg_mae_MVAr": round(float(np.abs(Qg - qg_true).mean()) * base, 2),
        "Pg_sum_pred_MW": round(float(Pg.sum()) * base, 1),
        "Pg_sum_true_MW": round(float(pg_true.sum()) * base, 1),
        "load_MW": round(pd_tot * base, 1),
        "cost_true": round(c_true, 1), "cost_pred": round(c_pred, 1),
        "cost_err_%": round(100 * (c_pred - c_true) / max(abs(c_true), 1e-9), 2),
        "feas_prob": round(float(out["feas"]), 3),
        "V_pred_range": [round(float(V.min()), 3), round(float(V.max()), 3)],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--ckpt_dir", default="data/models/gridsfm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from huggingface_hub import hf_hub_download
    from gridsfm import load_model
    torch.set_num_threads(max(1, torch.get_num_threads()))
    p = hf_hub_download("microsoft/GridSFM_Open", CKPT, local_dir=a.ckpt_dir)
    model = load_model(p, device="cpu")
    model.eval()
    res = [evaluate(model, Path(s)) for s in a.scenarios]
    for r in res:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
