"""Shared HTTP + raw-cache helpers. Every download is recorded in a manifest
(url without secrets, timestamp, sha256, size) so data/raw can be re-fetched."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("abgrid.collect")

SECRET_PARAMS = {"api-key", "apikey", "key", "token"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(raw_dir: Path, file: Path, url: str, params: dict | None = None, note: str = ""):
    """Append an entry to data/raw/MANIFEST.jsonl. Never stores headers/keys."""
    safe = {k: v for k, v in (params or {}).items() if k.lower() not in SECRET_PARAMS}
    entry = {
        "file": str(file.relative_to(raw_dir)),
        "url": url,
        "params": safe,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": file.stat().st_size,
        "sha256": sha256(file),
        "note": note,
    }
    with open(raw_dir / "MANIFEST.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def get(url: str, *, params=None, headers=None, timeout=120, retries=4, method="GET", data=None):
    last = None
    for i in range(retries):
        try:
            r = requests.request(method, url, params=params, headers=headers,
                                 timeout=timeout, data=data)
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last = e
            wait = 5 * 2**i
            log.warning("request failed (%s), retry %d in %ds", type(e).__name__, i + 1, wait)
            time.sleep(wait)
    raise last


def aeso_headers(cfg: dict) -> dict:
    """The key comes from the env var (or is injected by the network proxy).
    It is never logged or written to disk."""
    a = cfg["sources"]["aeso"]
    return {a["key_header"]: os.environ.get(a["key_env"], "proxy-injected")}
