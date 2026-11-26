"""Exercise the relaxation hierarchy: scale load (and generation headroom is
fixed) until AC-OPF needs relaxations; record the level reached per factor.
This is the 'quality signal' test for Phase 4."""
from __future__ import annotations

import copy

import pandapower as pp

from .opf import ac_opf, sanity


def load_scaling(cfg, net_path, factors=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5)):
    base = pp.from_json(net_path)
    hv = base.load.name.astype(str).str.startswith("hvdc_")
    rows = []
    for f in factors:
        n = copy.deepcopy(base)
        n.load.loc[~hv, "p_mw"] *= f
        n.load.loc[~hv, "q_mvar"] *= f
        sol, rec = ac_opf(n, cfg)
        row = {"load_factor": f, "load_mw": round(float(n.load.p_mw[~hv].sum()), 0),
               "feasible": rec["feasible"], "relaxation_level": rec["relaxation_level"],
               "relaxation_name": rec.get("relaxation_name"),
               "seconds_total": round(sum(a["seconds"] for a in rec["attempts"]), 1)}
        if rec["feasible"]:
            s = sanity(sol, ac=True)
            row.update({k: s[k] for k in ("losses_%", "load_shed_mw", "lmp_max", "vm_min", "flags")})
        rows.append(row)
    return rows
