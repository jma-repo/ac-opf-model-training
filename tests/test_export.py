"""Export tests: the three formats agree, OPFData column widths, and a
MATPOWER round trip reproduces the AC-OPF objective."""
import json
from pathlib import Path

import numpy as np
import pytest

from abgrid.config import load_config

CFG = load_config()
ROOT = Path(CFG["_root"])
SOL = ROOT / "outputs" / "phase3" / "ac_solution_net.json"
pytestmark = pytest.mark.skipif(not SOL.exists(), reason="base case not solved")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    import pandapower as pp
    from abgrid.export import case as ex
    net = pp.from_json(SOL)
    d = tmp_path_factory.mktemp("exp")
    c = ex.write_all(net, d, "t", solved=True)
    return net, c, d


def test_opfdata_widths(exported):
    _, _, d = exported
    g = json.loads((d / "t.pyg.json").read_text())["grid"]
    assert {len(r) for r in g["nodes"]["bus"]} == {4}
    assert {len(r) for r in g["nodes"]["generator"]} == {11}
    assert {len(r) for r in g["edges"]["ac_line"]["features"]} == {9}
    assert {len(r) for r in g["edges"]["transformer"]["features"]} == {11}


def test_formats_agree(exported):
    _, c, d = exported
    pm = json.loads((d / "t.powermodels.json").read_text())
    g = json.loads((d / "t.pyg.json").read_text())["grid"]
    base = c["baseMVA"]
    x_pm = np.array([pm["branch"][str(i + 1)]["br_x"] for i in range(c["n_line"])])
    x_g = np.array([f[5] for f in g["edges"]["ac_line"]["features"]])
    assert np.allclose(x_pm, x_g) and np.allclose(x_g, c["branch"][: c["n_line"], 3])
    pmax_pm = np.array([pm["gen"][str(i + 1)]["pmax"] for i in range(len(c["gen"]))])
    assert np.allclose(pmax_pm * base, c["gen"][:, 8])
    assert np.isclose(sum(v["pd"] for v in pm["load"].values()) * base, c["load_pq"][:, 0].sum())


def test_per_unit_ranges(exported):
    _, c, _ = exported
    br = c["branch"]
    assert (br[:, 3] > 0).all() and (br[:, 3] < 5).all()      # x in pu, not ohm
    assert (br[:, 2] >= 0).all()
    assert (c["bus"][:, 11] > c["bus"][:, 12]).all()          # vmax > vmin
    assert (c["bus"][:, 1] == 3).sum() == 1                   # exactly one reference bus


def test_matpower_roundtrip_objective(exported):
    _, c, d = exported
    from abgrid.export.roundtrip import roundtrip
    r = roundtrip(d / "t.m", c)
    assert r["n_bus"] == len(c["bus"]) and r["n_branch"] == r["n_branch_original"]
    assert r["n_units"] == r["n_gen_original"]
    assert r["objective_rel_diff"] < 5e-3
    assert abs(r["total_pg_roundtrip"] - r["total_pg_original"]) < 0.01 * r["total_pg_original"]


def test_relaxation_hierarchy_records_level():
    import pandapower as pp
    from abgrid.solve.opf import ac_opf
    net = sorted((ROOT / "data/interim/params").glob("net_*.json"))[-1]
    n = pp.from_json(net)
    hv = n.load.name.astype(str).str.startswith("hvdc_")
    n.load.loc[~hv, ["p_mw", "q_mvar"]] *= 1.2      # known to need level 1
    _, rec = ac_opf(n, CFG)
    assert rec["feasible"] and rec["relaxation_level"] >= 1
    assert rec["attempts"][0]["name"] == "nominal" and not rec["attempts"][0]["converged"]
