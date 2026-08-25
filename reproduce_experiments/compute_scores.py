import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 5 noise cells in canonical paper order: Clean -> Annot -> Struct -> Mixed -> Worst.
# Display labels match the paper (`tab:all_per_cell` column headers); directory
# IDs match `_runner.py` and the canonical archive.
USER_SCOPE_CELLS = [
    ("Clean", "eps_0.0_rho_0.0_ri_0.0"),
    ("Annot", "eps_0.0_rho_0.0_ri_0.3"),
    ("Struct", "eps_0.3_rho_1.0_ri_0.0"),
    ("Mixed", "eps_0.3_rho_0.5_ri_0.3"),
    ("Worst", "eps_0.3_rho_1.0_ri_0.3"),
]

SYNTH_METHODS = [
    "mtdnn", "gce", "bootstrapping", "coteaching", "evidential",
    "moe_gate", "pooled_ce", "pooled_bootstrapping",
    "if_bls", "fqbert", "mtlnl", "excessmtl", "ifs_hesitant",
]
BLITZER_METHODS = [
    "mtdnn", "gce", "forward_correction", "bootstrapping", "coteaching",
    "evidential", "moe_gate", "pooled_ce", "pooled_bootstrapping",
    "if_bls", "fqbert", "mtlnl", "excessmtl", "ifs_hesitant",
]
BERT_METHODS = [
    "mtdnn", "gce", "bootstrapping", "coteaching", "evidential",
    "moe_gate", "pooled_ce", "pooled_bootstrapping",
    "fqbert", "mtlnl", "excessmtl", "ifs_hesitant",
]

# Internal method id -> paper display label.
DISPLAY_LABEL = {
    "mtdnn": "MT-DNN",
    "gce": "GCE",
    "forward_correction": "Forward Corr.",
    "bootstrapping": "Bootstrap.",
    "coteaching": "Co-teach.",
    "evidential": "Evidential",
    "moe_gate": "MoE-gate",
    "pooled_ce": "Pooled-CE",
    "pooled_bootstrapping": "Pooled-Boot.",
    "if_bls": "IF-BLS",
    "fqbert": "FQ-IFS",
    "mtlnl": "MTL-NL",
    "excessmtl": "ExcessMTL",
    "ifs_hesitant": "IFS-Hesitant",
}

BERT_THIRD_TASK = {"phase3_berta": "rte", "phase4_bertb": "sst2"}

# Task order per phase (matters for indexing into the per-task `mu`/`nu`/`alpha`
# lists in the IFS / Evidential eval JSONs).
BERT_TASK_ORDER = {
    "phase3_berta": ["mrpc", "rte", "cola"],
    "phase4_bertb": ["mrpc", "sst2", "cola"],
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "reproduce"

# (K, B) score pair. None for either component if disk data missing.
KBPair = Tuple[Optional[float], Optional[float]]


def read_history_kb(cell_dir: Path) -> KBPair:
    f = cell_dir / "history.json"
    if not f.exists():
        return (None, None)
    history = json.loads(f.read_text())
    if not history:
        return (None, None)
    last = history[-1]
    k = round(last["acc_known"] * 100, 2) if "acc_known" in last else None
    b = round(last["acc_blind"] * 100, 2) if "acc_blind" in last else None
    return (k, b)


def _mcc(yt, yp) -> float:
    yt, yp = np.asarray(yt, int), np.asarray(yp, int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def _bert_split_score(memb, labels, tasks, lo: int, hi: int,
                      third_task: str, task_order: List[str],
                      slc: str) -> Optional[float]:
    tti = {t: i for i, t in enumerate(task_order)}
    by = defaultdict(list)
    for i in range(lo, hi):
        tk = tasks[i];
        yt = labels[i]
        row = memb[i]
        if slc == "K" and isinstance(row.get("mu"), list) and isinstance(row.get("nu"), list):
            ti = tti.get(tk, 0)
            yp = 1 if row["mu"][ti] > row["nu"][ti] else 0
        elif slc == "K" and isinstance(row.get("alpha"), list):
            ti = tti.get(tk, 0)
            a = row["alpha"][ti]
            yp = 1 if a[1] > a[0] else 0
        else:
            yp = int(row.get("predicted_label", 0))
        by[tk].append((yt, yp))
    out = {}
    for tk in ["cola", "mrpc", "rte", "sst2"]:
        if tk not in by:
            continue
        yt = [a for a, _ in by[tk]]
        yp = [b for _, b in by[tk]]
        out[tk] = _mcc(yt, yp) if tk == "cola" else float(np.mean(np.array(yt) == np.array(yp)))
    if not all(k in out for k in ["cola", "mrpc", third_task]):
        return None
    return round((out["cola"] + out["mrpc"] + out[third_task]) / 3 * 100, 2)


def read_bert_kb(cell_dir: Path, phase: str, epoch: int = 9) -> KBPair:
    m = cell_dir / f"eval_memberships_epoch_{epoch}.json"
    l = cell_dir / f"eval_labels_epoch_{epoch}.csv"
    t = cell_dir / f"eval_task_names_epoch_{epoch}.txt"
    if not (m.exists() and l.exists() and t.exists()):
        return (None, None)
    labels = [int(x) for x in l.read_text().splitlines() if x.strip()]
    tasks = [x.strip().lower() for x in t.read_text().splitlines() if x.strip()]
    memb = json.loads(m.read_text())
    if not (len(labels) == len(tasks) == len(memb)):
        return (None, None)
    third_task = BERT_THIRD_TASK[phase]
    task_order = BERT_TASK_ORDER[phase]
    n = len(labels);
    third = n // 3
    k = _bert_split_score(memb, labels, tasks, 0, third, third_task, task_order, "K")
    b = _bert_split_score(memb, labels, tasks, third, 2 * third, third_task, task_order, "B")
    return (k, b)


def collect_history_scale(scale_dir: Path, methods: List[str],
                          cells: List[Tuple[str, str]]) -> Dict[str, Dict[str, KBPair]]:
    out: Dict[str, Dict[str, KBPair]] = {}
    for m in methods:
        row: Dict[str, KBPair] = {}
        for cell_name, cell_id in cells:
            cd = scale_dir / m / cell_id
            row[cell_name] = read_history_kb(cd) if cd.exists() else (None, None)
        out[m] = row
    return out


def collect_bert_scale(scale_dir: Path, methods: List[str], phase: str
                       ) -> Dict[str, Dict[str, KBPair]]:
    out: Dict[str, Dict[str, KBPair]] = {}
    for m in methods:
        row: Dict[str, KBPair] = {}
        for cell_name, cell_id in USER_SCOPE_CELLS:
            cd = scale_dir / m / cell_id
            row[cell_name] = (read_bert_kb(cd, phase)
                              if cd.exists() else (None, None))
        out[m] = row
    return out


def aggregate_scale(scale: str, seeds: List[int]) -> Tuple[List[str], List[Tuple[str, str]],
Dict[int, Dict[str, Dict[str, KBPair]]]]:
    if scale == "synthetic":
        methods, cells = SYNTH_METHODS, USER_SCOPE_CELLS
        per_seed = {s: collect_history_scale(RESULTS_ROOT / "synthetic" / f"seed_{s}",
                                             methods, cells) for s in seeds}
    elif scale == "blitzer":
        methods, cells = BLITZER_METHODS, USER_SCOPE_CELLS
        per_seed = {s: collect_history_scale(RESULTS_ROOT / "blitzer" / f"seed_{s}",
                                             methods, cells) for s in seeds}
    elif scale.startswith("bert_"):
        methods, cells = BERT_METHODS, USER_SCOPE_CELLS
        phase = scale.removeprefix("bert_")
        per_seed = {s: collect_bert_scale(RESULTS_ROOT / scale / f"seed_{s}",
                                          methods, phase) for s in seeds}
    else:
        raise ValueError(f"Unknown scale: {scale}")
    return methods, cells, per_seed


def _fmt_single(v: Optional[float]) -> str:
    return "  --  " if v is None else f"{v:5.1f}"


def _fmt_mean_std(vals: List[float]) -> str:
    if not vals:
        return "  --  "
    if len(vals) == 1:
        return f"{vals[0]:5.1f}"
    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
    return f"{mean:5.1f}+/-{std:.1f}"


def print_scale_table(scale: str, seeds: List[int]) -> None:
    methods, cells, per_seed = aggregate_scale(scale, seeds)
    cell_names = [c[0] for c in cells]

    # 10-column header: <Cell> K, <Cell> B per cell.
    print(f"\n## {scale}\n")
    header_parts = ["Method"]
    for n in cell_names:
        header_parts.extend([f"{n} K", f"{n} B"])
    print("| " + " | ".join(header_parts) + " |")
    print("|" + "|".join("---" for _ in header_parts) + "|")

    single = len(seeds) == 1

    for m in methods:
        label = DISPLAY_LABEL.get(m, m)
        entries: List[str] = []
        if single:
            s = seeds[0]
            for n in cell_names:
                k, b = per_seed[s][m][n]
                entries.append(_fmt_single(k))
                entries.append(_fmt_single(b))
        else:
            for n in cell_names:
                ks = [per_seed[s][m][n][0] for s in seeds if per_seed[s][m][n][0] is not None]
                bs = [per_seed[s][m][n][1] for s in seeds if per_seed[s][m][n][1] is not None]
                entries.append(_fmt_mean_std(ks))
                entries.append(_fmt_mean_std(bs))
        print("| " + label + " | " + " | ".join(entries) + " |")


ALL_SCALES = [
    "synthetic",  # tab:all_per_cell block (a)
    "blitzer",  # tab:all_per_cell block (b)
    "bert_phase3_berta",  # tab:all_per_cell block (c) -- BERT Cfg A
    "bert_phase4_bertb",  # tab:all_per_cell block (d) -- BERT Cfg B
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=ALL_SCALES + ["all"], default="all",
                    help="Which scale's results to aggregate.")
    ap.add_argument("--seed", type=int,
                    help="Single seed (shortcut for --seeds N).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Multiple seeds; per-cell mean+/-std is reported.")
    args = ap.parse_args()

    if args.seed is not None and args.seeds is not None:
        ap.error("Pass either --seed or --seeds, not both.")
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 42]

    scales = ALL_SCALES if args.scale == "all" else [args.scale]
    for s in scales:
        # Skip silently if the scale's directory doesn't exist (not yet run).
        if not any((RESULTS_ROOT / s / f"seed_{seed}").exists() for seed in seeds):
            continue
        print_scale_table(s, seeds)


if __name__ == "__main__":
    main()
