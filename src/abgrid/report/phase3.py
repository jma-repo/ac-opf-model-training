"""Phase 3 report: parameter defaults, provenance of electrical/gen/load
attributes, and the base-case DC/AC-OPF sanity checks."""
from __future__ import annotations

import json

import pandas as pd

from .. import provenance as pv
from ..config import path


def run(cfg):
    d = path(cfg, "interim_dir") / "params"
    out = path(cfg, "output_dir") / "phase3"
    t = {"branch_e": pd.read_csv(d / "branches.csv"), "transformer_e": pd.read_csv(d / "transformers.csv"),
         "generator": pd.read_csv(d / "generators.csv"), "load": pd.read_csv(d / "loads.csv")}
    prov = pv.summary(t)
    prov.to_csv(out / "provenance_summary.csv", index=False)
    # whole-model summary: phase 2 + phase 3
    p2 = pd.read_csv(path(cfg, "output_dir") / "phase2" / "provenance_summary.csv")
    allp = pd.concat([p2, prov], ignore_index=True)
    allp.to_csv(path(cfg, "output_dir") / "provenance_report.csv", index=False)
    kinds = allp[["open_%", "derived_%", "synthetic_%"]].mean().round(1)

    summ = json.loads(next(d.glob("params_summary_*.json")).read_text())
    solve = json.loads((out / "solve_summary.json").read_text())
    p = cfg["params"]
    ld = pd.DataFrame(p["line_defaults"]).T.rename_axis("kV class").reset_index()
    td = pd.DataFrame(p["transformer_defaults"]).T.rename_axis("pair").reset_index()
    fu = pd.DataFrame(p["fuels"]).T.rename_axis("fuel").reset_index()
    ac, dc = solve["ac"], solve["dc"]
    lines = [
        "# Phase 3 parameters and base case", "",
        "> Approximate open-data model, NOT AESO's grid. Cost curves are illustrative only.", "",
        f"Snapshot: **{summ['snapshot_mpt']} MPT**, {summ['season_ratings']} ratings.", "",
        "## Defaults used where open data is missing", "",
        "Line R and B (X and ratings come from CODERS). R = X_open / (X/R of the class).", "",
        ld.to_markdown(index=False), "",
        "Sources/assumptions: Kundur, *Power System Stability and Control*, Table 6.1 (230/500 kV); "
        "typical 266/477 kcmil ACSR for 69/138 kV (Glover, Sarma & Overbye, App. A).",
        f"Open CODERS X per km (median by class, ohm/km): {summ['x_ohm_per_km_median_by_kv']} "
        "— consistent with the textbook range, which supports using open X directly.", "",
        "Transformers (synthetic by voltage pair; rating raised to "
        f"{p['transformer_sizing_factor']}×min(HV, LV connected line ratings) where larger; "
        f"{summ['transformers_upsized']} upsized):", "", td.to_markdown(index=False), "",
        "Generator cost and reactive limits (synthetic, illustrative):", "", fu.to_markdown(index=False), "",
        "## Generators and load", "",
        f"- Active units: {summ['generators_active']} (dropped {summ['generators_dropped_inactive']} "
        "retired/not yet built). Converted coal-to-gas units re-typed as gas_steam using the arXiv/AESO type.",
        f"- Installed MW by fuel: {summ['installed_mw_by_fuel']}",
        f"- Available MW at snapshot (wind/solar from AESO metered volume): {summ['available_mw_by_fuel']}",
        f"- Load: {summ['load_mw']} MW on {summ['load_buses']} buses "
        f"(AESO planning-area {p['load_column']}; unplaced {summ['load_unplaced_mw']} MW), pf {p['load_power_factor']}.",
        "- HVDC WATL/EATL: fixed schedule "
        f"{cfg['topology']['hvdc_schedule_mw']} MW north→south, {cfg['topology']['hvdc_loss_percent']}% loss.",
        f"- Interties: fixed net schedule {p['intertie_schedule_mw']} MW.", "",
        "## Base-case OPF", "",
        "| | DC-OPF | AC-OPF |", "|---|---|---|",
        f"| converged | {dc['converged']} | {ac['feasible']} (relaxation level {ac.get('relaxation_level')}) |",
        f"| solve time s | {dc['seconds']} | {ac['attempts'][-1]['seconds']} |",
        f"| generation MW (excl. HVDC) | {dc.get('gen_mw')} | {ac.get('gen_mw')} |",
        f"| AC losses MW (%) | — | {ac.get('losses_mw')} ({ac.get('losses_%')}%) |",
        f"| cost (illustrative CAD/h) | {dc.get('cost')} | {ac.get('cost')} |",
        f"| max line loading (S, %) | {dc.get('max_line_loading_%')} | {ac.get('max_line_loading_%')} |",
        f"| voltage range pu | — | {ac.get('vm_min')}–{ac.get('vm_max')} |",
        f"| LMP min/median/max | {dc.get('lmp_min')}/{dc.get('lmp_median')}/{dc.get('lmp_max')} | "
        f"{ac.get('lmp_min')}/{ac.get('lmp_median')}/{ac.get('lmp_max')} |",
        f"| flags | {dc.get('flags')} | {ac.get('flags')} |", "",
        f"DC-OPF median LMP by kV class: {dc.get('lmp_by_kv_median')}", "",
        "## Provenance (Phase 3 attributes)", "", prov.drop(columns="sources").to_markdown(index=False), "",
        f"Mean share over all tracked attributes (Phases 2+3): open {kinds['open_%']}%, "
        f"derived {kinds['derived_%']}%, synthetic {kinds['synthetic_%']}%. "
        "Full table: `outputs/provenance_report.csv`.",
    ]
    (out / "phase3_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
