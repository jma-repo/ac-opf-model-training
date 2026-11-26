"""Topology builder tests: connectivity, no dangling buses, units, provenance.
Run after `python -m abgrid.cli build-topology` (uses data/interim)."""
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from abgrid import provenance as pv
from abgrid.config import load_config

CFG = load_config()
D = Path(CFG["_root"]) / CFG["project"]["interim_dir"] / "topology"
pytestmark = pytest.mark.skipif(not (D / "buses.csv").exists(), reason="topology not built")


@pytest.fixture(scope="module")
def t():
    return {n: pd.read_csv(D / f"{f}.csv") for n, f in
            [("bus", "buses"), ("branch", "branches"), ("transformer", "transformers"),
             ("dc_line", "dc_lines"), ("boundary", "boundary")]}


def test_connected(t):
    G = nx.Graph()
    G.add_nodes_from(t["bus"].bus)
    G.add_edges_from(zip(t["branch"].from_bus, t["branch"].to_bus))
    G.add_edges_from(zip(t["transformer"].hv_bus, t["transformer"].lv_bus))
    assert nx.is_connected(G)


def test_no_dangling_references(t):
    buses = set(t["bus"].bus)
    for name, cols in [("branch", ["from_bus", "to_bus"]), ("transformer", ["hv_bus", "lv_bus"]),
                       ("dc_line", ["from_bus", "to_bus"]), ("boundary", ["bus"])]:
        for c in cols:
            assert set(t[name][c]) <= buses, f"{name}.{c} references unknown bus"


def test_every_bus_has_an_edge(t):
    used = set(t["branch"].from_bus) | set(t["branch"].to_bus) | set(t["transformer"].hv_bus) \
        | set(t["transformer"].lv_bus)
    assert set(t["bus"].bus) <= used


def test_no_self_loops(t):
    assert (t["branch"].from_bus != t["branch"].to_bus).all()


def test_branch_ends_share_voltage_class(t):
    kv = t["bus"].set_index("bus").base_kv
    b = t["branch"]
    assert (kv[b.from_bus].values == kv[b.to_bus].values).all()
    assert (kv[b.from_bus].values == b.kv_class.values).all()


def test_units_and_ranges(t):
    b, bus = t["branch"], t["bus"]
    assert set(bus.base_kv) <= {69, 138, 240, 500}
    assert (b.voltage_kv >= CFG["topology"]["min_kv"]).all()
    assert b.length_km.between(0, 1000).all()              # km, not m
    assert (b.x_ohm >= 0).all() and (b.x_ohm < 1000).all()  # ohm, not pu
    assert (b.rate_summer_mw > 0).all() and (b.rate_summer_mw < 5000).all()
    assert bus.lat.between(48.9, 60.2).all() and bus.lon.between(-120.2, -109.8).all()


def test_transformers_step_down(t):
    tr = t["transformer"]
    assert (tr.hv_kv > tr.lv_kv).all()
    kv = t["bus"].set_index("bus").base_kv
    assert (kv[tr.hv_bus].values == tr.hv_kv.values).all()


def test_one_bus_per_node_and_class(t):
    assert not t["bus"].duplicated(["node_code", "base_kv"]).any()


def test_provenance_complete(t):
    import geopandas as gpd
    for name, df in t.items():
        if name == "branch":   # geometry lives in the GeoJSON
            df = gpd.read_file(D / "branches.geojson")
        pv.validate(df, name)


def test_provenance_validator_rejects_bad_flags():
    df = pd.DataFrame({"bus": [1], "bus__src": ["guess"], "neighbour": ["BC"],
                       "neighbour__src": ["open:x"], "import_max_mw": [1],
                       "import_max_mw__src": ["open:x"], "export_max_mw": [1],
                       "export_max_mw__src": ["open:x"]})
    with pytest.raises(ValueError):
        pv.validate(df, "boundary")
