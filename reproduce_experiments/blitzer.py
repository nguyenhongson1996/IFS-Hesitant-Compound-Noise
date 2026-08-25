import argparse

from reproduce_experiments._runner import (
    USER_SCOPE_CELLS, cell_dir_name, out_root, run_method,
)

# (paper_label, runner_module, extra_args).
# blitzer_grid handles most baselines via --method; IF-BLS and FQ-BERT use their own dedicated runners.
METHODS = [
    ("mtdnn", "experiments.blitzer_grid", ["--method", "mtdnn"]),
    ("gce", "experiments.blitzer_grid", ["--method", "gce"]),
    ("forward_correction", "experiments.blitzer_grid", ["--method", "forward_correction"]),
    ("bootstrapping", "experiments.blitzer_grid", ["--method", "bootstrapping"]),
    ("coteaching", "experiments.blitzer_grid", ["--method", "coteaching", "--warmup_epochs", "0"]),
    ("evidential", "experiments.blitzer_grid", ["--method", "evidential"]),
    ("moe_gate", "experiments.blitzer_grid", ["--method", "task_identity"]),
    ("pooled_ce", "experiments.blitzer_grid", ["--method", "pooled_ce"]),
    ("pooled_bootstrapping", "experiments.blitzer_grid", ["--method", "pooled_bootstrapping"]),
    ("if_bls", "experiments.baseline_if_bls", []),
    ("fqbert", "experiments.blitzer_fq", []),
    ("mtlnl", "experiments.blitzer_grid", ["--method", "mtlnl"]),
    ("excessmtl", "experiments.blitzer_grid", ["--method", "excessmtl"]),
    # IFS-Hesitant reproduction config.
    ("ifs_hesitant", "experiments.blitzer_grid",
     ["--method", "ifs_hesitant", "--head_type", "factored", "--bayes_trust",
      "--weight_norm", "none", "--l_pred_alpha", "0.0"]),
]

# Blitzer protocol: 15 epochs, batch 64, lr 1e-3, TF-IDF max_features=2000,
# ngram in {1,2}, small-MLP 64->32, warm-up 5. w_min=0 for small-arch
# (paper Sec. 6: floor inactive on small encoders).
COMMON_ARGS = [
    "--dataset", "blitzer",
    "--num_epochs", "15", "--batch_size", "64", "--lr", "1e-3",
    "--max_features", "2000", "--ngram_max", "2", "--min_df", "5",
    "--encoder_hidden", "64", "--encoder_output", "32",
    "--warmup_epochs", "5", "--w_min", "0",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = out_root("blitzer", args.seed)
    n_total = len(METHODS) * len(USER_SCOPE_CELLS)
    n_ok = 0

    for method_label, module, extras in METHODS:
        for cell_name, eps, rho, ri in USER_SCOPE_CELLS:
            out_dir = root / method_label / cell_dir_name(eps, rho, ri)
            log_path = root / "logs" / f"{method_label}_{cell_name}.log"
            # IF-BLS and FQ-BERT runners have their own minimal flag surface.
            if method_label == "if_bls":
                extras_full = list(extras) + ["--dataset", "blitzer"]
            elif method_label == "fqbert":
                extras_full = list(extras) + ["--num_epochs", "15", "--batch_size", "64",
                                              "--lr", "1e-3", "--warmup_epochs", "5"]
            else:
                extras_full = list(extras) + COMMON_ARGS
            ok = run_method(
                runner_module=module,
                extra_args=extras_full,
                eps=eps, rho=rho, ri=ri,
                seed=args.seed,
                out_dir=out_dir,
                log_path=log_path,
            )
            n_ok += int(ok)

    print(f"\n=== Blitzer seed {args.seed}: {n_ok}/{n_total} cells succeeded ===")
    print(f"Results under: {root}")

    print(f"\n=== Paper tab:all_per_cell block (b) - Blitzer - seed {args.seed} ===")
    from reproduce_experiments.compute_scores import print_scale_table
    print_scale_table(scale="blitzer", seeds=[args.seed])


if __name__ == "__main__":
    main()
