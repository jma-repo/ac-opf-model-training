"""CODERS (SESIT / Energy Modelling Hub) API. The key is a `key` URL query
parameter, read from env var CODERS_API_KEY. HTTPS only. The key is scrubbed
from every log line, exception message and manifest entry."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

from .common import record

log = logging.getLogger("abgrid.collect.coders")


class CodersError(RuntimeError):
    pass


def _key(cfg) -> str:
    k = os.environ.get(cfg["sources"]["coders"]["key_env"], "")
    if not k:
        raise CodersError("CODERS_API_KEY not set; skipping CODERS (see README)")
    return k


def _get(cfg, endpoint: str, params: dict | None = None):
    base = cfg["sources"]["coders"]["base_url"]
    if not base.startswith("https://"):
        raise CodersError("CODERS base_url must be https (key travels in the URL)")
    key = _key(cfg)
    q = dict(params or {}, key=key)
    try:
        r = requests.get(f"{base}/{endpoint.lstrip('/')}", params=q, timeout=300)
    except requests.RequestException as e:
        raise CodersError(str(e).replace(key, "***")) from None
    if r.status_code != 200:
        raise CodersError(f"CODERS /{endpoint}: HTTP {r.status_code}")
    return f"{base}/{endpoint.lstrip('/')}", r.json()


def fetch(cfg: dict, raw_root: Path):
    c = cfg["sources"]["coders"]
    out = raw_root / "coders"
    out.mkdir(parents=True, exist_ok=True)
    url, tables = _get(cfg, "tables")
    f = out / "tables.json"
    f.write_text(json.dumps(tables, indent=1))
    record(raw_root, f, url)                       # params never include the key
    log.info("CODERS tables: %s", tables)
    for t in c["tables"]:
        url, data = _get(cfg, t, c.get("filters", {}).get(t, {}))
        f = out / f"{t}.json"
        f.write_text(json.dumps(data))
        n = len(data) if isinstance(data, list) else "?"
        record(raw_root, f, url, c.get("filters", {}).get(t, {}), note=f"{n} records")
        log.info("CODERS %s: %s records", t, n)
