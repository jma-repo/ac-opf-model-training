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
    sub.add_parser("build-topology")
    sub.add_parser("report-phase2")
    sub.add_parser("build-params")
    sub.add_parser("solve-base")
    sub.add_parser("report-phase3")
    g5 = sub.add_parser("scenarios")
    g5.add_argument("--n", type=int, default=None)
    g5.add_argument("--run", default=None)
    g5.add_argument("--procs", type=int, default=None)
    p4 = sub.add_parser("phase4")
    p4.add_argument("--skip-stress", action="store_true")
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
    elif args.cmd == "build-topology":
        from .conflate import build
        build.build(cfg)
    elif args.cmd == "build-params":
        from .params import build as pbuild
        pbuild.build(cfg)
    elif args.cmd == "solve-base":
        import json as _j
        from .solve import opf
        d = path(cfg, "interim_dir") / "params"
        net = sorted(d.glob("net_*.json"))[-1]
        out = path(cfg, "output_dir") / "phase3"
        out.mkdir(parents=True, exist_ok=True)
        print(_j.dumps(opf.solve(cfg, net, out), indent=1, default=float))
    elif args.cmd == "scenarios":
        from .scenarios import generate
        df, out = generate.generate(cfg, n=args.n, run_name=args.run, n_proc=args.procs)
        print(df.groupby("family")[["feasible", "ac_feasible_any", "dc_feasible"]].mean().round(3))
        print(df[["dc_s", "ac_s_nominal", "ac_s_total", "wall_s"]].describe().round(2))
    elif args.cmd == "phase4":
        from .report import phase4
        phase4.run(cfg, skip_stress=args.skip_stress)
    elif args.cmd == "report-phase3":
        from .report import phase3
        phase3.run(cfg)
    elif args.cmd == "report-phase2":
        from .report import phase2
        phase2.run(cfg)
    elif args.cmd == "report-phase1":
        from .report import phase1
        phase1.run(cfg)


if __name__ == "__main__":
    main()
