import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from reproduce_experiments._runner import USER_SCOPE_CELLS, cell_dir_name, out_root

# Each method has its own runner. (paper_label, runner_module, extra_args).
METHODS = [
    ("mtdnn", "experiments.baseline_mtdnn", ["--lambda_other", "1.0"]),
    ("gce", "experiments.baseline_bert_robust_loss", ["--loss_type", "gce"]),
    ("bootstrapping", "experiments.baseline_bert_robust_loss", ["--loss_type", "bootstrapping"]),
    ("coteaching", "experiments.baseline_coteaching", ["--warmup_epochs", "0", "--lambda_other", "1.0"]),
    ("evidential", "experiments.baseline_bert_evidential", ["--kl_anneal_epochs", "10"]),
    ("moe_gate", "experiments.baseline_task_identity", ["--lambda_other", "1.0"]),
    ("pooled_ce", "experiments.baseline_pooled", ["--loss_type", "ce", "--lambda_other", "1.0"]),
    ("pooled_bootstrapping", "experiments.baseline_pooled", ["--loss_type", "bootstrapping", "--lambda_other", "1.0"]),
    ("fqbert", "experiments.baseline_fqbert", ["--lambda_hes", "1.0"]),
    ("mtlnl", "experiments.baseline_bert_mtlnl_excessmtl", ["--method", "mtlnl"]),
    ("excessmtl", "experiments.baseline_bert_mtlnl_excessmtl", ["--method", "excessmtl"]),
    # IFS-Hesitant reproduction config.
    ("ifs_hesitant", "experiments.noisy_mtl_ifs",
     ["--head_type", "ifs", "--bayes_trust",
      "--w_min", "0.1", "--weight_norm", "none",
      "--l_pred_alpha", "0.0"]),
]

CONFIGS = {
    "A": ("phase3_berta", ["MRPC", "RTE", "COLA"]),
    "B": ("phase4_bertb", ["MRPC", "SST2", "COLA"]),
}

# BERT protocol (paper Sec. 6 implementation defaults).
COMMON_ARGS = [
    "--num_epochs", "10", "--batch_size", "16",
    "--max_gradient_clip", "1.0",
    "--scheduler_type", "linear", "--warmup_ratio", "0.0",
    "--weight_decay", "0.0",
]


def run_bert(*, runner_module, extra_args, tasks, eps, rho, ri,
             seed, out_dir, log_path):
    if (out_dir / ".done").exists():
        print(f"[SKIP] {out_dir}", flush=True);
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", runner_module, *extra_args,
        "--epsilon_t", str(eps), "--class_noise_rho", str(rho),
        "--class_noise_rho_indep", str(ri),
        "--noise_seed", str(seed),
        "--tasks", *tasks,
        "--out_dir", str(out_dir),
        *COMMON_ARGS,
    ]
    t0 = time.time()
    print(f"[RUN ] {time.strftime('%H:%M:%S')} {runner_module} -> {out_dir}",
          flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
    elapsed = time.time() - t0
    if rc == 0:
        (out_dir / ".done").touch()
        print(f"[ OK ] {time.strftime('%H:%M:%S')} ({elapsed:.0f}s)", flush=True)
        return True
    print(f"[FAIL] {out_dir} (see {log_path})", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    configs = ["A", "B"] if args.config == "both" else [args.config]
    n_total = 0
    n_ok = 0

    for cfg in configs:
        cfg_label, tasks = CONFIGS[cfg]
        root = out_root(f"bert_{cfg_label}", args.seed)
        for method_label, module, extras in METHODS:
            for cell_name, eps, rho, ri in USER_SCOPE_CELLS:
                n_total += 1
                out_dir = root / method_label / cell_dir_name(eps, rho, ri)
                log_path = root / "logs" / f"{method_label}_{cell_name}.log"
                ok = run_bert(
                    runner_module=module, extra_args=extras,
                    tasks=tasks,
                    eps=eps, rho=rho, ri=ri, seed=args.seed,
                    out_dir=out_dir, log_path=log_path,
                )
                n_ok += int(ok)

    print(f"\n=== BERT-GLUE seed {args.seed}: {n_ok}/{n_total} cells succeeded ===")

    # Display aggregated table(s).
    from reproduce_experiments.compute_scores import print_scale_table
    for cfg in configs:
        block = {"A": "(c)", "B": "(d)"}[cfg]
        scale = f"bert_{'phase3_berta' if cfg == 'A' else 'phase4_bertb'}"
        print(f"\n=== Paper tab:all_per_cell block {block} (BERT Cfg {cfg}) - seed {args.seed} ===")
        print_scale_table(scale=scale, seeds=[args.seed])


if __name__ == "__main__":
    main()
