"""Command line entry point:  python -m abgrid.cli collect [--only goa,osm,aeso]"""
from __future__ import annotations

import argparse
import logging

from .config import load_config, path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abgrid")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--only", default="goa,osm,aeso,gen_capacity,aeso_files,coders")
    sub.add_parser("report-phase1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    raw = path(cfg, "raw_dir")

    if args.cmd == "collect":
        only = set(args.only.split(","))
        if "goa" in only:
            from .collect import goa_powerlines
            goa_powerlines.fetch(cfg, raw)
        if "osm" in only:
            from .collect import osm
            osm.fetch(cfg, raw)
        if "aeso" in only:
            from .collect import aeso
            aeso.fetch(cfg, raw)
        if "gen_capacity" in only:
            from .collect import aeso
            aeso.fetch_gen_capacity(cfg, raw)
        if "coders" in only:
            from .collect import coders
            try:
                coders.fetch(cfg, raw)
            except coders.CodersError as e:
                logging.getLogger("abgrid").warning("CODERS skipped: %s", e)
        if "aeso_files" in only:
            from .collect import aeso_files
            aeso_files.fetch(cfg, raw)
    elif args.cmd == "report-phase1":
        from .report import phase1
        phase1.run(cfg)


if __name__ == "__main__":
    main()
