"""Static files from AESO's public Data Requests page (planning-area hourly
load, planning-area boundaries). Downloaded only; never executed."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from .common import get, record

log = logging.getLogger("abgrid.collect.aeso_files")


def fetch(cfg: dict, raw_root: Path):
    s = cfg["sources"]["aeso_files"]
    out = raw_root / "aeso_files"
    out.mkdir(parents=True, exist_ok=True)
    for key, rel in s["files"].items():
        url = s["base_url"] + rel
        f = out / Path(rel).name
        if not f.exists():
            r = get(url, headers={"User-Agent": "abgrid-research/0.1"}, timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
            record(raw_root, f, url, note=key)
        if f.suffix == ".zip":
            dest = out / f.stem
            with zipfile.ZipFile(f) as z:
                for m in z.namelist():   # guard against path traversal
                    if m.startswith("/") or ".." in Path(m).parts:
                        raise ValueError(f"unsafe zip member {m}")
                z.extractall(dest)
        log.info("%s: %s (%d bytes)", key, f.name, f.stat().st_size)
