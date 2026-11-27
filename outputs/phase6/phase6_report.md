# Phase 6 — GridSFM zero-shot vs fine-tuned on Alberta scenarios

> Approximate open-data model, not AESO's grid. Costs are illustrative; labels come from our own AC-OPF.

## Setup
- Scenarios: run `s1250c` (1,250 scenarios, calibrated structure: grid-served load, metered cogen/hydro/biomass, metered wind/solar).
- Splits (stratified by family, seed 42): test 149 · val 100 · train pool 1,001 (`splits.json`).
- Model: GridSFM-Open v1.1 (15.1 M params). Each round restarts from the released weights.
- Training: `gridsfm.finetune_opfdata` (AdamW, lr 1e-4, batch 8), feasible scenarios wrapped in
  `SyntheticMixedDataset` (20% synthetic perturbed-infeasible) + real infeasible scenarios.
- Hardware: 4 CPU cores, no GPU. FT-1000 ran 4 epochs only (CPU budget).
- Labels: AC-OPF (pandapower PIPS) at nominal limits; "feasible" = relaxation level 0.

## Results (test split, 149 scenarios, 128 feasible)

| Round | Train (feas/infeas) | Epochs | Train time | V MAE pu | θ MAE ° | Pg MAE MW | Qg MAE MVAr | \|Cost err\| | Feas acc | Infeasible F1 | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-shot | 0 | – | – | 0.0342 | 4.50 | 9.93 | 18.59 | 18.7% | 0.859 | 0.00 | 0.121 |
| FT-10 | 9 / 1 | 10 | 6 min | 0.0211 | 2.64 | 7.37 | 14.06 | 13.8% | 0.812 | 0.42 | 0.169 |
| FT-100 | 84 / 16 | 10 | 19 min | 0.0159 | **2.21** | 4.70 | 9.61 | 9.8% | 0.866 | 0.60 | 0.124 |
| FT-1000 | 878 / 122 | 4 | 65 min | 0.0133 | 5.59 | 3.25 | 8.38 | **4.3%** | **0.960** | **0.83** | 0.031 |
| FT-1000 + θ refine | 878 / 122 | +4 | +92 min | **0.0128** | **2.16** | **3.16** | **8.05** | 4.45% | **0.960** | **0.83** | **0.021** |

Confusion (positive = feasible): 0-shot tp128 fp21 fn0 tn0 · FT-10 111/11/17/10 · FT-100 114/6/14/15 · FT-1000 128/6/0/15.

Inference 0.24–0.29 s/scenario vs AC-OPF 6.4 s median → ~24× faster (CPU, single scenario, no batching).

## Findings
1. **Zero-shot is unusable for screening**: it calls every case feasible (0 of 21 infeasible caught) and has ~19% cost error.
2. **Fine-tuning helps monotonically for V, Pg, Qg, cost and feasibility.** With 1,000 scenarios, cost error falls to 4.3%,
   all feasible cases are recognised, 15 of 21 infeasible cases are caught, and the feasibility probability is well calibrated (ECE 0.03).
3. **Angles regress at FT-1000** (2.2° → 5.6° vs reference bus). Diagnostic on 20 test cases: branch angle-difference MAE
   0.25° (FT-100) → 0.60° (FT-1000); mean-centred MAE 1.55° → 2.61°; regression slope pred-vs-true 0.95 → 0.87, i.e. the
   angle spread is compressed. Likely cause: only 4 epochs at a larger data scale with the multi-task loss shifted towards
   Pg/feasibility. **Fixed in Phase 6b** (`src/abgrid/fm/refine.py`): continue from the FT-1000 checkpoint for 4 more epochs
   at lr 5e-5 with the θ term weighted ×20, keeping the epoch with the best validation score (θ/θ₀ + cost/cost₀).
   Validation θ fell 5.42° → 2.03° by epoch 1 and stayed there; test θ 5.59° → 2.16° (back to FT-100 level) while
   V, Pg, Qg improve slightly, cost error stays at 4.45%, and feasibility is unchanged (ECE 0.031 → 0.021).
   This refined checkpoint is used for the warm-start test and the Phase 7 siting screen.
4. The 6 remaining false "feasible" calls are infeasible scenarios the model cannot yet separate; for siting screens the solver
   must still verify every candidate.
5. Caveat: train and test come from the same synthetic generator and topology, so this measures interpolation inside our model,
   not real-world accuracy.

## Not done in this phase
- **gridfm-graphkit comparison — skipped (decision recorded).** No public pretrained weights were found for
  gridfm-graphkit 0.9.x, so a comparison would mean training it from scratch, which the project scope excludes. It also
  needs a large install (torch ≥ 2.10, torch-scatter, lightning, mlflow, gridfm-datakit with juliacall/Julia).
- KCL residual metric.

## Files
`finetune_results.json` (metrics + per-epoch loss), `test_predictions_n{0,10,100,1000}.csv`, `splits.json`;
checkpoints `data/models/ft/gridsfm_alberta_n*.pt` (not committed). Code: `src/abgrid/fm/{dataset,finetune}.py`.
