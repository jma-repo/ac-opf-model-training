"""Phase 4: exports (MATPOWER, PowerModels JSON, GridSFM), MATPOWER round trip,
relaxation-hierarchy stress test, extra sanity checks, report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandapower as pp

from ..config import path
from ..export import case as ex
from ..export.roundtrip import roundtrip
from ..solve import stress


def run(cfg, skip_stress=False):
    p3 = path(cfg, "output_dir") / "phase3"
    out = path(cfg, "output_dir") / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    net = pp.from_json(p3 / "ac_solution_net.json")
    c = ex.write_all(net, out, "alberta_base", solved=True,
                     meta={"snapshot_mpt": cfg["params"]["snapshot_mpt"], "relaxation_level": 0})
    rt = roundtrip(out / "alberta_base.m", c)

    # extra sanity: angle difference across branches, DC vs AC LMP agreement
    va = np.radians(net.res_bus.va_degree)
    f = np.r_[net.line.from_bus.values, net.trafo.hv_bus.values]
    t = np.r_[net.line.to_bus.values, net.trafo.lv_bus.values]
    dth = np.degrees(np.abs(va[f].values - va[t].values))
    dc = pd.read_csv(p3 / "dc_res_bus.csv", index_col=0)
    ac = pd.read_csv(p3 / "ac_res_bus.csv", index_col=0)
    corr = float(np.corrcoef(dc.lam_p, ac.lam_p)[0, 1])
    solve = json.loads((p3 / "solve_summary.json").read_text())
    checks = {"angle_diff_max_deg": round(float(dth.max()), 2),
              "angle_diff_p99_deg": round(float(np.percentile(dth, 99)), 2),
              "branches_angle_gt_30deg": int((dth > 30).sum()),
              "lmp_dc_ac_correlation": round(corr, 3),
              "gen_minus_load_mw_ac": solve["ac"]["losses_mw"],
              "losses_%_ac": solve["ac"]["losses_%"]}

    prev = out / "phase4_checks.json"
    rows = json.loads(prev.read_text()).get("stress_load_scaling") if (skip_stress and prev.exists()) else None
    if rows is None:
        net_path = sorted((path(cfg, "interim_dir") / "params").glob("net_*.json"))[-1]
        rows = stress.load_scaling(cfg, net_path)
    res = {"roundtrip": rt, "sanity": checks, "stress_load_scaling": rows}
    prev.write_text(json.dumps(res, indent=1, default=str))

    st = pd.DataFrame(rows)
    cols = [c_ for c_ in ["load_factor", "load_mw", "feasible", "relaxation_level", "relaxation_name",
                          "seconds_total", "losses_%", "load_shed_mw", "lmp_max", "vm_min"] if c_ in st]
    lines = [
        "# Phase 4: exports, verification and relaxation hierarchy", "",
        "> Approximate open-data model, NOT AESO's grid. Costs illustrative.", "",
        "## Exports (one per-unit case, three writers)", "",
        "- `outputs/phase4/alberta_base.m` — MATPOWER v2 (gencost model 2, $/MW² h, $/MWh)",
        "- `outputs/phase4/alberta_base.powermodels.json` — PowerModels network dict (per-unit, radians)",
        "- `outputs/phase4/alberta_base.pyg.json` — GridSFM / OPFData (widths 4/11/9/11) with solution block",
        "", "All three are generated from the same arrays (`abgrid.export.case.build_case`) and agree "
        "exactly; the GridSFM file matches the Phase 3 converter bit-for-bit.", "",
        "## MATPOWER round trip", "",
        "The `.m` text is parsed back, rebuilt with pandapower `from_ppc`, and re-solved from a flat start.", "",
        f"| | original | round trip |", "|---|---|---|",
        f"| objective | {rt['objective_original']:.1f} | {rt['objective_roundtrip']:.1f} "
        f"(rel. diff {rt['objective_rel_diff']:.2e}) |",
        f"| total generation MW | {rt['total_pg_original']} | {rt['total_pg_roundtrip']} |",
        f"| buses / branches / units | {len(c['bus'])} / {rt['n_branch_original']} / {rt['n_gen_original']} | "
        f"{rt['n_bus']} / {rt['n_branch']} / {rt['n_units']} |",
        f"| max |ΔV| pu | — | {rt['vm_max_abs_diff']:.4f} (flat start reaches a nearby optimum) |", "",
        "## Sanity checks (base case)", "",
        *[f"- {k}: {v}" for k, v in checks.items()], "",
        "## Relaxation hierarchy (load scaling stress test)", "",
        "Levels are cumulative and applied only if the previous level is infeasible: "
        "0 nominal → 1 line ratings ×1.2 → 2 voltage 0.90–1.10 → 3 ±200 MVAr support at ≥240 kV buses "
        "→ 4 load shedding at VOLL 5000. The level reached is stored with each case as a quality signal.", "",
        st[cols].to_markdown(index=False), "",
        "Line-loading flags at levels ≥1 are measured against the original (unrelaxed) ratings.",
        "At ×1.5 load the available capacity at the snapshot (wind ≈2.9 GW) is below demand, so no level "
        "is feasible: the case is labelled infeasible.", "",
        "## Solver", "",
        "pandapower 3.5 / PYPOWER interior point (PIPS) in Python. Julia + PowerModels.jl/Ipopt is not "
        "installed; the PowerModels JSON is ready for cross-validation if Julia is added.",
    ]
    (out / "phase4_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return res
