"""Phase 6b: fix the FT-1000 angle regression.

GridSFM's compute_loss has no theta weight, and theta enters as a squared
error in radians (5 deg -> 0.0076), small next to the other terms. We wrap
compute_loss to add (theta_weight - 1) * L_theta (same anchored, capped
definition as GridSFM), continue from the FT-1000 checkpoint at a lower lr,
and after each epoch keep the checkpoint with the best validation score
    theta_mae / theta_mae_start + cost_err / cost_err_start.
Run in .venv-fm:  python -I src/abgrid/fm/refine.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from abgrid.config import load_config  # noqa: E402
from abgrid.fm.dataset import AlbertaScenarios, make_transform, train_dataset  # noqa: E402
from abgrid.fm.finetune import CKPT, evaluate  # noqa: E402


def theta_loss(batch):
    from gridsfm.blocks import _slack_or_mean_anchor
    from gridsfm.loss import PER_ELEM_CAP, _per_graph_mean, build_feasible_masks
    m = build_feasible_masks(batch)
    gmask, bus_mask = m["gmask"], m["bus_mask"]
    if bus_mask is None or not bus_mask.any():
        return batch["bus"].pred.new_zeros(())
    bp, bt = batch["bus"].pred, batch["bus"].y
    ng = int(getattr(batch, "num_graphs", 1) or 1)
    ref = _slack_or_mean_anchor(batch, bt[:, 0], n_graphs=ng)
    d = torch.atan2(torch.sin(bp[bus_mask, 0] - ref[bus_mask]), torch.cos(bp[bus_mask, 0] - ref[bus_mask]))
    sq = PER_ELEM_CAP * torch.tanh(d.pow(2) / PER_ELEM_CAP)
    return _per_graph_mean(sq, batch["bus"].batch[bus_mask], int(gmask.numel()))[gmask].mean()


def patch_loss(weight: float):
    import importlib
    fo = importlib.import_module("gridsfm.finetune_opfdata")
    if not hasattr(fo, "compute_loss"):   # package re-exports the function under the module name
        fo = sys.modules["gridsfm.finetune_opfdata"]
    orig = fo.compute_loss

    def wrapped(batch, **kw):
        L, parts = orig(batch, **kw)
        return L + (weight - 1.0) * theta_loss(batch), parts
    fo.compute_loss = wrapped


def main():
    from torch_geometric.loader import DataLoader
    from gridsfm import load_model
    from gridsfm.finetune_opfdata import finetune_opfdata
    cfg = load_config(); f = cfg["finetune"]; r = f["refine"]
    sdir = ROOT / cfg["project"]["interim_dir"] / "scenarios" / f["scenario_run"]
    labels = pd.read_csv(sdir / "labels.csv").set_index("id", drop=False)
    P = lambda i: sdir / "pyg" / f"scn_{int(i):05d}.pyg.json"  # noqa: E731
    out = ROOT / "outputs" / "phase6"
    sp = json.loads((out / "splits.json").read_text())
    torch.manual_seed(f["seed"]); torch.set_num_threads(f["threads"])
    tf = make_transform()
    patch_loss(r["theta_weight"])
    ids = sp["train_pool"][: r["n_train"]]
    tr = train_dataset([P(i) for i in ids if labels.loc[i, "feasible"]],
                       [P(i) for i in ids if not labels.loc[i, "feasible"]], tf,
                       infeas_prob=f["infeas_prob"], seed=f["seed"] + 1)
    val_ds = AlbertaScenarios([P(i) for i in sp["val"]], transform=tf)
    val_paths = [P(i) for i in sp["val"]]
    m = load_model(CKPT, device="cpu")
    start = ROOT / "data/models/ft" / f"gridsfm_alberta_n{r['n_train']}.pt"
    m.load_state_dict(torch.load(start)["state_dict"])
    v0, _ = evaluate(m, val_paths, labels, tf)
    hist = [{"epoch": -1, "val": v0, "score": 2.0}]
    best = {"score": 2.0, "epoch": -1}
    ck = ROOT / "data/models/ft" / f"gridsfm_alberta_n{r['n_train']}_refined.pt"
    torch.save({"state_dict": m.state_dict()}, ck)
    print("start val", v0, flush=True)

    def on_end(e):
        v, _ = evaluate(m, val_paths, labels, tf); m.train()
        s = v["theta_mae_deg"] / v0["theta_mae_deg"] + v["cost_err_abs_pct"] / v0["cost_err_abs_pct"]
        hist.append({"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e.get("val_loss"),
                     "val": v, "score": round(s, 3)})
        if s < best["score"]:
            best.update(score=s, epoch=e["epoch"]); torch.save({"state_dict": m.state_dict()}, ck)
        print(f"epoch {e['epoch']} train {e['train_loss']:.4f} score {s:.3f} theta {v['theta_mae_deg']} "
              f"cost {v['cost_err_abs_pct']} f1 {v['infeasible_f1']} {e['epoch_s']}s", flush=True)

    t0 = time.time()
    finetune_opfdata(m, DataLoader(tr, batch_size=f["batch_size"], shuffle=True),
                     val_loader=DataLoader(val_ds, batch_size=f["batch_size"]),
                     epochs=r["epochs"], lr=r["lr"], on_epoch_end=on_end)
    m.load_state_dict(torch.load(ck)["state_dict"])
    met, df = evaluate(m, [P(i) for i in sp["test"]], labels, tf)
    df.to_csv(out / f"test_predictions_n{r['n_train']}_refined.csv", index=False)
    res = {"refine": r, "best_epoch": best["epoch"], "train_s": round(time.time() - t0, 1),
           "history": hist, "test": met}
    (out / "refine_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("refined test", met, flush=True)


if __name__ == "__main__":
    main()
