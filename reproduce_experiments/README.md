# Paper Reproduction — One Script per Dataset Block

Each script trains all methods from scratch for one seed and writes
results under `results/reproduce/<scale>/seed_<n>/<method>/<cell>/`.
`.done` markers make every script idempotent; finished cells are
skipped on re-run. After the run, the script prints the paper-format
table via `compute_scores`.

Together, the four scripts reproduce **`tab:all_per_cell`**, the full
per-cell appendix table spanning the four dataset blocks used in the
paper's experimental story.

| Paper table block | Driver | Scale | Approx time (1 seed, single GPU RTX 4070) |
|---|---|---|-------------------------------------------|
| `tab:all_per_cell` (a) | `python -m reproduce_experiments.synthetic --seed 42` | 3 binary tasks in R^5, MLP 64->32 | ~10 min                                   |
| `tab:all_per_cell` (b) | `python -m reproduce_experiments.blitzer --seed 42` | 4-domain sentiment, TF-IDF + MLP 64->32 | ~30 min                                   |
| `tab:all_per_cell` (c) | `python -m reproduce_experiments.bert_glue --seed 42 --config A` | MRPC+RTE+CoLA, BERT-base bs=16 | ~30 h                                     |
| `tab:all_per_cell` (d) | `python -m reproduce_experiments.bert_glue --seed 42 --config B` | MRPC+SST-2+CoLA, BERT-base bs=16 | ~40 h                                     |

Every row of `tab:all_per_cell` reports 10 numeric values: K (task-known,
with verified task identity at test time) and B (blind task-agnostic,
model-routed) for each of the 5 noise cells (Clean, Annot, Struct,
Mixed, Worst). The aggregator prints those 10 values per row.

The headline Mixed/Worst summaries and the derived per-cell analyses
are extracted from the per-scale tables above, so there is no separate
training driver for them in this compact reproduction flow.

## Multi-seed (paper-grade n=5, seeds 42-46)

```bash
for s in 42 43 44 45 46; do
    python -m reproduce_experiments.synthetic  --seed $s
    python -m reproduce_experiments.blitzer    --seed $s
    python -m reproduce_experiments.bert_glue  --seed $s
done

python -m reproduce_experiments.compute_scores --seeds 42 43 44 45 46
```

The final aggregator prints all 4 blocks of `tab:all_per_cell` with
per-cell mean +/- std across seeds, in the paper's row order, with
display labels matching the paper.

## IFS-Hesitant headline configuration

Used in every driver's `ifs_hesitant` row:

```
--head_type {ifs|factored} --bayes_trust --l_pred_alpha 0.0 --weight_norm none
  --w_min 0.1 (BERT) | --w_min 0 (small-arch)
```

Implements the IFS factored head with the Bayes-corrected trust weight
`w_i = (1-pi_t_tilde_i) * sat_tilde_t_tilde_i(y_tilde_i)` and no L_pred
branch (paper §4). `--head_type ifs` (BERT) and `--head_type factored`
(small-arch) refer to the same parameterisation on different runners.

## Output layout

```
results/reproduce/<scale>/seed_<n>/
    <method>/
        eps_<eps>_rho_<rho_c>_ri_<rho_indep>/
            .done                       (success marker)
            history.json / eval_*.json  (per-runner output)
    logs/
        <method>_<cell_name>.log
```

## Aggregator standalone

If results already exist, print without re-training:

```bash
python -m reproduce_experiments.compute_scores --scale synthetic --seed 42
python -m reproduce_experiments.compute_scores --scale bert_phase4_bertb --seeds 42 43 44 45 46
python -m reproduce_experiments.compute_scores --seeds 42 43 44 45 46         # all 4 blocks
```

## Baselines (paper row order)

| Script label          | Paper row     | Scales |
|-----------------------|---------------|--------|
| `mtdnn`               | MT-DNN        | all    |
| `gce`                 | GCE           | all    |
| `forward_correction`  | Forward Corr. | Blitzer only |
| `bootstrapping`       | Bootstrap.    | all    |
| `coteaching`          | Co-teach.     | all (warm-up 0, faithful Han 2018) |
| `evidential`          | Evidential    | all    |
| `moe_gate`            | MoE-gate      | all (K column uses task-known eval with verified task identity) |
| `pooled_ce`           | Pooled-CE     | all    |
| `pooled_bootstrapping`| Pooled-Boot.  | all    |
| `if_bls`              | IF-BLS        | Synthetic, Blitzer (closed-form, no BERT) |
| `fqbert`              | FQ-IFS        | all    |
| `mtlnl`               | MTL-NL        | all    |
| `excessmtl`           | ExcessMTL     | all    |
| `ifs_hesitant` *(ours)* | IFS-Hesitant | all    |

## Reproducing from an Existing Results Archive (no re-train)

If you already have the archived `results/` tree, you can re-print the
paper-style rows without re-training by running `compute_scores`
directly on the existing outputs.

## Environment

```
torch 2.4.0  |  transformers 4.48.3  |  datasets 3.5.0
```

If `transformers` cannot find `BatchEncoding`, pin to `4.48.3`.
