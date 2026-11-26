"""Round-trip validation: parse the exported MATPOWER .m text back, rebuild a
pandapower net with from_ppc, re-solve AC-OPF and compare objective, dispatch
and voltages against the original solution."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_m(path: Path) -> dict:
    txt = Path(path).read_text()
    out = {"baseMVA": float(re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)", txt).group(1)), "version": "2"}
    for key in ("bus", "gen", "branch", "gencost"):
        body = re.search(rf"mpc\.{key}\s*=\s*\[(.*?)\];", txt, re.S).group(1)
        rows = [r.split() for r in body.strip().split(";") if r.strip()]
        out[key] = np.array(rows, dtype=float)
    return out


def roundtrip(m_path: Path, original_case: dict) -> dict:
    import pandapower as pp
    from pandapower.converter.pypower import from_ppc
    ppc = parse_m(m_path)
    net = from_ppc(ppc, f_hz=60, validate_conversion=False)
    pp.runopp(net, init="flat", verbose=False, calculate_voltage_angles=True, OPF_FLOW_LIM="S")
    obj = float(net.res_cost)
    # from_ppc maps ratio-1 transformers to `impedance` and some gens to
    # `sgen`/`ext_grid`; count all of them
    pg_total = float(net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum()
                     + (net.res_sgen.p_mw.sum() if len(net.sgen) else 0.0))
    n_units = len(net.gen) + len(net.sgen) + len(net.ext_grid)
    vm_new = net.res_bus.vm_pu.values
    oc = original_case
    return {
        "objective_original": oc["objective"], "objective_roundtrip": round(obj, 2),
        "objective_rel_diff": abs(obj - oc["objective"]) / abs(oc["objective"]),
        "total_pg_original": round(float(oc["gen"][:, 1].sum()), 1), "total_pg_roundtrip": round(pg_total, 1),
        "vm_max_abs_diff": float(np.abs(vm_new - oc["bus"][:, 7]).max()),
        "n_bus": len(net.bus), "n_units": n_units, "n_gen_original": len(oc["gen"]),
        "n_branch": len(net.line) + len(net.trafo) + len(net.impedance),
        "n_branch_original": len(oc["branch"]),
    }
