"""AESO public API (apimgw.aeso.ca/public). Key read from env var or injected
by the network proxy; never logged. Only documented endpoints are used."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .common import aeso_headers, get, record

log = logging.getLogger("abgrid.collect.aeso")


def _call(cfg, ep, params, timeout=300):
    a = cfg["sources"]["aeso"]
    url = a["base_url"] + a["endpoints"][ep]
    r = get(url, params=params, headers=aeso_headers(cfg), timeout=timeout)
    if r.status_code in (401, 403):
        raise PermissionError(f"AESO {ep}: HTTP {r.status_code} (check API key)")
    r.raise_for_status()
    return url, r.json()["return"]


def _windows(start: str, end: str, days: int):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s <= e:
        t = min(s + timedelta(days=days - 1), e)
        yield s.isoformat(), t.isoformat()
        s = t + timedelta(days=1)


def fetch(cfg: dict, raw_root: Path):
    a = cfg["sources"]["aeso"]
    out = raw_root / "aeso"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Asset list
    url, data = _call(cfg, "assetlist", {})
    f = out / "assetlist.json"
    f.write_text(json.dumps(data))
    record(raw_root, f, url, note=f"{len(data)} assets")
    log.info("asset list: %d", len(data))

    # 2. Alberta Internal Load, hourly (chunked by ~quarter)
    rows = []
    for s, e in _windows(a["ail_start"], a["ail_end"], 92):
        _, d = _call(cfg, "ail", {"startDate": s, "endDate": e})
        rows.extend(d["Actual Forecast Report"])
        log.info("AIL %s..%s: %d rows", s, e, len(d["Actual Forecast Report"]))
    f = out / "ail_hourly.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    record(raw_root, f, a["base_url"] + a["endpoints"]["ail"],
           {"startDate": a["ail_start"], "endDate": a["ail_end"], "chunk_days": 92},
           note=f"{len(rows)} hourly rows")

    # 3. Metered volume, all assets (max 16 days per call)
    for w in a["metered_volume_windows"]:
        recs = []
        for s, e in _windows(w[0], w[1], 16):
            url, d = _call(cfg, "metered_volume", {"startDate": s, "endDate": e}, timeout=600)
            for pp in d:
                for asset in pp["asset_list"]:
                    for mv in asset["metered_volume_list"]:
                        recs.append((pp["pool_participant_ID"], asset["asset_ID"],
                                     asset["asset_class"], mv["begin_date_utc"],
                                     mv["metered_volume"]))
        df = pd.DataFrame(recs, columns=["pool_participant_ID", "asset_ID", "asset_class",
                                         "begin_date_utc", "metered_volume_mwh"])
        f = out / f"metered_volume_{w[0]}_{w[1]}.csv.gz"
        df.to_csv(f, index=False)
        record(raw_root, f, url, {"startDate": w[0], "endDate": w[1]},
               note=f"{len(df)} rows, {df.asset_ID.nunique()} assets")
        log.info("metered volume %s..%s: %d rows, %d assets", w[0], w[1], len(df),
                 df.asset_ID.nunique())


def fetch_gen_capacity(cfg: dict, raw_root: Path):
    """AIES Gen Capacity: hourly MC/AC/outage MW aggregated by fuel type."""
    a = cfg["sources"]["aeso"]
    out = raw_root / "aeso"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for s, e in _windows(a["gen_capacity_start"], a["gen_capacity_end"], 31):
        url, d = _call(cfg, "gen_capacity", {"startDate": s, "endDate": e})
        for ft in d:
            for h in ft["Hours"]:
                recs.append({"fuel_type": ft["fuel_type"], "sub_fuel_type": ft.get("sub_fuel_type"),
                             "begin_datetime_utc": h["begin_datetime_utc"],
                             **{k.replace(" ", "_"): v for k, v in h["outage_grouping"].items()}})
        log.info("gen capacity %s..%s", s, e)
    df = pd.DataFrame(recs)
    f = out / "aies_gen_capacity_hourly.csv.gz"
    df.to_csv(f, index=False)
    record(raw_root, f, url, {"startDate": a["gen_capacity_start"], "endDate": a["gen_capacity_end"],
                              "chunk_days": 31}, note=f"{len(df)} rows")
    return f


def fetch_metered_assets(cfg: dict, raw_root: Path, asset_ids, start: str, end: str,
                         name: str = "renewables"):
    """Hourly metered volume for selected assets (one call per asset; the
    by-asset endpoint returns ~100K records/call, enough for a year)."""
    out = raw_root / "aeso"
    rows, missing = [], []
    url = None
    for aid in sorted(set(asset_ids)):
        url, d = _call(cfg, "metered_volume", {"startDate": start, "endDate": end, "asset_ID": aid},
                       timeout=600)
        got = False
        for pp_ in d:
            for a in pp_["asset_list"]:
                if a["asset_ID"] != aid:
                    continue
                got = True
                rows.extend((aid, m["begin_date_mpt"], m["metered_volume"]) for m in a["metered_volume_list"])
        if not got:
            missing.append(aid)
    df = pd.DataFrame(rows, columns=["asset_ID", "begin_date_mpt", "metered_volume_mwh"])
    f = out / f"metered_{name}_{start}_{end}.csv.gz"
    df.to_csv(f, index=False)
    record(raw_root, f, url, {"startDate": start, "endDate": end, "assets": len(set(asset_ids))},
           note=f"{len(df)} rows, {df.asset_ID.nunique()} assets, missing {len(missing)}")
    log.info("metered %s: %d rows, %d assets, %d missing", name, len(df), df.asset_ID.nunique(), len(missing))
    return f, missing
