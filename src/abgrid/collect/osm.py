"""OpenStreetMap power infrastructure for Alberta via Overpass, queried in
bounding-box tiles with a pause between calls (rate-limit friendly).
Falls back across configured endpoints. Licence: ODbL."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

import requests

from .common import record

log = logging.getLogger("abgrid.collect.osm")

QUERY = """[out:json][timeout:{timeout}];
(
  way["power"~"^(line|minor_line|cable)$"]({s},{w},{n},{e});
  way["power"~"^(substation|plant|generator)$"]({s},{w},{n},{e});
  node["power"~"^(substation|plant|generator)$"]({s},{w},{n},{e});
  relation["power"~"^(substation|plant)$"]({s},{w},{n},{e});
);
out body geom qt;
"""


def tiles(bbox, step):
    w, s, e, n = bbox
    for lat in np.arange(s, n, step[1]):
        for lon in np.arange(w, e, step[0]):
            yield (round(lon, 4), round(lat, 4),
                   round(min(lon + step[0], e), 4), round(min(lat + step[1], n), 4))


def _query(cfg, q):
    """Try each endpoint; on HTTP 429 wait progressively longer (polite)."""
    o = cfg["sources"]["osm"]
    err = None
    for wait in [0] + list(o["rate_limit_wait_s"]):
        time.sleep(wait)
        for ep in o["overpass_endpoints"]:
            try:
                r = requests.post(ep, data={"data": q}, timeout=o["timeout_s"] + 60)
                if r.status_code == 200:
                    return ep, r.json()
                err = RuntimeError(f"{ep}: HTTP {r.status_code}")
            except requests.RequestException as ex:  # try next endpoint
                err = ex
            log.warning("endpoint failed: %s", err)
    raise err


def fetch(cfg: dict, raw_root: Path):
    o = cfg["sources"]["osm"]
    out = raw_root / "osm" / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    for i, (w, s, e, n) in enumerate(tiles(cfg["region"]["bbox"], cfg["region"]["tile_deg"])):
        f = out / f"tile_{i:02d}_{w}_{s}.json"
        if f.exists():
            continue
        q = QUERY.format(timeout=o["timeout_s"], s=s, w=w, n=n, e=e)
        ep, data = _query(cfg, q)
        f.write_text(json.dumps(data))
        record(raw_root, f, ep, {"bbox_swne": [s, w, n, e]},
               note=f"{len(data.get('elements', []))} elements")
        log.info("tile %d (%s,%s): %d elements", i, w, s, len(data.get("elements", [])))
        time.sleep(o["pause_s"])
