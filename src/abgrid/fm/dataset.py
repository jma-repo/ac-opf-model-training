"""Alberta scenarios -> GridSFM fine-tuning HeteroData (OPFData schema).

Each Phase 5 `.pyg.json` is read with GridSFM's own `load_opfdata` (which
validates OPFData column widths), then labelled like PyG's OPFDataset:
  bus.y       [Nb, 2] = (va rad, vm pu)
  generator.y [Ng, 2] = (pg, qg) pu
  edge_label  [E, 4]  = (pf, qf, pt, qt) pu  for ac_line and transformer
  feasible    0/1     (AC-OPF feasible at nominal limits, relaxation level 0)
Infeasible scenarios carry zero labels; GridSFM's loss masks them out of the
regression terms and uses them for the feasibility head."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset


def load_labelled(path: Path):
    from gridsfm.data import load_opfdata
    d = json.loads(Path(path).read_text())
    g = load_opfdata(path)
    feas = bool(d["metadata"].get("feasible")) and "solution" in d
    nb, ng = g["bus"].x.size(0), g["generator"].x.size(0)
    if feas:
        s = d["solution"]
        g["bus"].y = torch.tensor(s["nodes"]["bus"], dtype=torch.float32)
        g["generator"].y = torch.tensor(s["nodes"]["generator"], dtype=torch.float32)
        for et in ("ac_line", "transformer"):
            key = ("bus", et, "bus")
            if key in g.edge_types:
                g[key].edge_label = torch.tensor(s["edges"][et]["features"], dtype=torch.float32)
    else:
        g["bus"].y = torch.zeros((nb, 2))
        g["generator"].y = torch.zeros((ng, 2))
        for et in ("ac_line", "transformer"):
            key = ("bus", et, "bus")
            if key in g.edge_types:
                g[key].edge_label = torch.zeros((g[key].edge_index.size(1), 4))
    g.feasible = torch.tensor(1 if feas else 0, dtype=torch.long)
    g.scenario_id = torch.tensor(int(d["metadata"].get("scenario_id", -1)))
    return g


class AlbertaScenarios(Dataset):
    """In-memory list of labelled graphs (982-bus graphs are small)."""

    def __init__(self, paths, transform=None):
        self.graphs = [load_labelled(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        from gridsfm import SyntheticMixedDataset
        g = self.graphs[i].clone()
        # same per-graph fields SyntheticMixedDataset attaches, so batches collate;
        # perturb_mode -2 marks a real (solver-labelled) scenario
        g.perturb_mode = torch.tensor(-2, dtype=torch.long)
        SyntheticMixedDataset._ensure_edge_active_mask(g)
        return self.transform(g) if self.transform is not None else g


def make_transform():
    from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
    from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_
    cb, pe = CycleBasisCache(), LaplacianFactorizationCache()

    def transform(data):
        prepare_for_grid_transformer_(data, cache=cb)
        attach_pe_features_(data, cache=pe)
        return data
    return transform


def train_dataset(feasible_paths, infeasible_paths, transform, infeas_prob=0.2, seed=42):
    """Feasible scenarios wrapped in GridSFM's SyntheticMixedDataset (which
    perturbs a fraction into synthetic infeasible graphs) + the real
    infeasible scenarios with their own labels."""
    from gridsfm import SyntheticMixedDataset
    parts = []
    if feasible_paths:
        parts.append(SyntheticMixedDataset(AlbertaScenarios(feasible_paths), infeas_prob=infeas_prob,
                                           seed=seed, transform=transform))
    if infeasible_paths:
        parts.append(AlbertaScenarios(infeasible_paths, transform=transform))
    return ConcatDataset(parts)
