# Compound Task-Class Noise in Task-Agnostic Inference for Multi-task Binary Classification: An Intuitionistic Fuzzy Approach

# Reproducing the Paper Experiments

This repository includes a compact reproduction flow under `reproduce_experiments/` for the paper on compound task-class noise in task-agnostic inference for multi-task binary classification.
Each driver runs one dataset block, writes outputs under `results/reproduce/`, and can be re-run safely because finished cells are marked with `.done`.
The same outputs reproduce the paper's per-cell result tables together with the derived Mixed/Worst summary views.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r torch-requirements.txt
```

If needed, set:

```bash
set PYTHONPATH=.
```

## Drivers

| Command | Paper block | Scale | Approx time              |
|---|---|---|--------------------------|
| `python -m reproduce_experiments.synthetic --seed 42` | `tab:all_per_cell` (a) | Synthetic | ~10 min                  |
| `python -m reproduce_experiments.blitzer --seed 42` | `tab:all_per_cell` (b) | Blitzer | ~30 min                  |
| `python -m reproduce_experiments.bert_glue --seed 42 --config A` | `tab:all_per_cell` (c) | BERT config A | ~30 h  (Using RTX 4070s) |
| `python -m reproduce_experiments.bert_glue --seed 42 --config B` | `tab:all_per_cell` (d) | BERT config B | ~40 h  (Using RTX 4070s) |

## Multi-seed run

```bash
for %s in (42 43 44 45 46) do python -m reproduce_experiments.synthetic --seed %s
for %s in (42 43 44 45 46) do python -m reproduce_experiments.blitzer --seed %s
for %s in (42 43 44 45 46) do python -m reproduce_experiments.bert_glue --seed %s

python -m reproduce_experiments.compute_scores --seeds 42 43 44 45 46
```

If you are using bash instead of Windows shell:

```bash
for s in 42 43 44 45 46; do
  python -m reproduce_experiments.synthetic --seed $s
  python -m reproduce_experiments.blitzer --seed $s
  python -m reproduce_experiments.bert_glue --seed $s
done

python -m reproduce_experiments.compute_scores --seeds 42 43 44 45 46
```

## Score aggregation

Use the aggregator after runs finish, or on existing results:

```bash
python -m reproduce_experiments.compute_scores --scale synthetic --seed 42
python -m reproduce_experiments.compute_scores --scale blitzer --seed 42
python -m reproduce_experiments.compute_scores --scale bert_phase3_berta --seed 42
python -m reproduce_experiments.compute_scores --scale bert_phase4_bertb --seed 42
python -m reproduce_experiments.compute_scores --seeds 42 43 44 45 46
```

`compute_scores` prints the paper-style 10-column rows for each method:

- `K` = task-known evaluation with verified task identity at test time
- `B` = blind task-agnostic evaluation, where the model must route the input itself
- Column order = Clean, Annot, Struct, Mixed, Worst

## Output layout

```text
results/reproduce/<scale>/seed_<n>/
  <method>/
    eps_<eps>_rho_<rho>_ri_<ri>/
      .done
      history.json
      eval_*.json
  logs/
    <method>_<cell>.log
```

## Noise cells

| Cell | Directory |
|---|---|
| Clean | `eps_0.0_rho_0.0_ri_0.0` |
| Annot | `eps_0.0_rho_0.0_ri_0.3` |
| Struct | `eps_0.3_rho_1.0_ri_0.0` |
| Mixed | `eps_0.3_rho_0.5_ri_0.3` |
| Worst | `eps_0.3_rho_1.0_ri_0.3` |

## Methods covered by the reproduction scripts

| Label | Paper row | Scales |
|---|---|---|
| `mtdnn` | MT-DNN | all |
| `gce` | GCE | all |
| `forward_correction` | Forward Corr. | Blitzer only |
| `bootstrapping` | Bootstrap. | all |
| `coteaching` | Co-teach. | all |
| `evidential` | Evidential | all |
| `moe_gate` | MoE-gate | all |
| `pooled_ce` | Pooled-CE | all |
| `pooled_bootstrapping` | Pooled-Boot. | all |
| `if_bls` | IF-BLS | Synthetic, Blitzer |
| `fqbert` | FQ-IFS | all |
| `mtlnl` | MTL-NL | all |
| `excessmtl` | ExcessMTL | all |
| `ifs_hesitant` | IFS-Hesitant | all |

## Headline IFS-Hesitant configuration used by the reproduction drivers

```text
--head_type {ifs|factored} --bayes_trust --l_pred_alpha 0.0 --weight_norm none
--w_min 0.1 (BERT) | --w_min 0 (MLP)
```

This is IFS-Hesitant setup used throughout the paper reproduction scripts.

## Environment

```text
torch 2.4.0
transformers 4.48.3
datasets 3.5.0
```

If `transformers` cannot find `BatchEncoding`, pin to `4.48.3`.

See `reproduce_experiments/README.md` for the per-driver breakdown.
