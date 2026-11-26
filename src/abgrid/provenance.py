"""Provenance tracking. Every model attribute column `a` has a companion
column `a__src` whose values are one of
    open:<dataset>      value taken directly from an open dataset
    derived:<method>    computed from open data by a documented method
    synthetic:<rule>    default / assumption, not from data
Tables declare their tracked attributes in SCHEMA; `validate` enforces it."""
from __future__ import annotations

import re

import pandas as pd

SRC_RE = re.compile(r"^(open|derived|synthetic):[A-Za-z0-9_.\-/+]+$")
KINDS = ("open", "derived", "synthetic")

# tracked attributes per element table (extended in later phases)
SCHEMA: dict[str, list[str]] = {
    "bus": ["name", "node_code", "lat", "lon", "base_kv", "node_type",
            "planning_area", "is_boundary"],
    "branch": ["from_bus", "to_bus", "circuit_id", "voltage_kv", "length_km",
               "n_circuits", "x_ohm", "rate_summer_mw", "rate_winter_mw", "geometry"],
    "transformer": ["hv_bus", "lv_bus", "hv_kv", "lv_kv"],
    "dc_line": ["from_bus", "to_bus", "p_max_mw", "name"],
    "boundary": ["bus", "import_max_mw", "export_max_mw", "neighbour"],
    # Phase 3 (electrical parameters)
    "branch_e": ["x_ohm_model", "r_ohm", "b_us", "length_model_km", "rate_mva"],
    "transformer_e": ["sn_mva", "vk_percent", "vkr_percent"],
    "generator": ["unit_code", "aeso_asset_id", "bus", "p_max_mw", "fuel", "availability",
                  "p_min_mw", "q_max_mvar", "q_min_mvar", "cost_c1", "cost_c2"],
    "load": ["bus", "planning_area", "p_mw", "q_mvar"],
}


def src(col: str) -> str:
    return f"{col}__src"


def put(df: pd.DataFrame, col: str, values, source) -> pd.DataFrame:
    """Set attribute `col` and its provenance (scalar or per-row series)."""
    df[col] = values
    df[src(col)] = source
    return df


def validate(df: pd.DataFrame, element: str) -> None:
    for a in SCHEMA[element]:
        if a not in df.columns:
            raise ValueError(f"{element}: missing attribute {a}")
        if src(a) not in df.columns:
            raise ValueError(f"{element}.{a}: missing provenance column")
        s = df[src(a)]
        bad = s[~s.astype(str).str.match(SRC_RE)]
        if len(bad):
            raise ValueError(f"{element}.{a}: invalid provenance values {bad.unique()[:5]}")


def summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for element, df in tables.items():
        for a in SCHEMA[element]:
            kind = df[src(a)].str.split(":").str[0]
            n = len(df)
            row = {"element": element, "attribute": a, "n": n}
            for k in KINDS:
                row[f"{k}_%"] = round(100 * (kind == k).sum() / n, 1) if n else 0.0
            row["sources"] = "; ".join(f"{k}={v}" for k, v in df[src(a)].value_counts().items())
            rows.append(row)
    return pd.DataFrame(rows)
