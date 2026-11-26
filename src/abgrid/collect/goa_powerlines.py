"""Government of Alberta Powerlines layer (MapServer/15). Polylines, NAD83.
No voltage / substation / status fields. Paged with resultOffset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .common import get, record

log = logging.getLogger("abgrid.collect.goa")


def fetch(cfg: dict, raw_root: Path) -> Path:
    src = cfg["sources"]["goa_powerlines"]
    out_dir = raw_root / "goa_powerlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = src["url"]
    meta = get(url.rsplit("/query", 1)[0], params={"f": "json"}).json()
    (out_dir / "layer_metadata.json").write_text(json.dumps(meta, indent=1))
    n = get(url, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}).json()["count"]
    log.info("GoA powerlines: %d records", n)

    feats, offset, page = [], 0, src["page_size"]
    while offset < n:
        params = {
            "where": "1=1", "outFields": "*", "outSR": 4269, "f": "geojson",
            "orderByFields": "OBJECTID ASC", "resultOffset": offset, "resultRecordCount": page,
        }
        r = get(url, params=params, timeout=180)
        r.raise_for_status()
        fc = r.json()
        got = fc.get("features", [])
        if not got:
            break
        feats.extend(got)
        offset += len(got)
        log.info("  %d / %d", offset, n)

    out = out_dir / "powerlines.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    record(raw_root, out, url, {"where": "1=1", "outSR": 4269, "f": "geojson",
                                "paged": f"resultOffset step {page}"},
           note=f"{len(feats)} of {n} features")
    if len(feats) != n:
        log.warning("expected %d features, got %d", n, len(feats))
    return out
