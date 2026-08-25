import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

# 5 user-scope cells: (cell_name, epsilon_t, class_noise_rho, class_noise_rho_indep).
USER_SCOPE_CELLS: List[Tuple[str, float, float, float]] = [
    ("clean", 0.0, 0.0, 0.0),
    ("annotator", 0.0, 0.0, 0.3),
    ("structured", 0.3, 1.0, 0.0),
    ("mixed", 0.3, 0.5, 0.3),
    ("worst", 0.3, 1.0, 0.3),
]


def cell_dir_name(eps: float, rho: float, ri: float) -> str:
    return f"eps_{eps}_rho_{rho}_ri_{ri}"


def run_method(
        *,
        runner_module: str,
        extra_args: Sequence[str],
        eps: float,
        rho: float,
        ri: float,
        seed: int,
        out_dir: Path,
        log_path: Path,
) -> bool:
    if (out_dir / ".done").exists():
        print(f"[SKIP] {out_dir}", flush=True)
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", runner_module,
        *extra_args,
        "--epsilon_t", str(eps),
        "--class_noise_rho", str(rho),
        "--class_noise_rho_indep", str(ri),
        "--noise_seed", str(seed),
        "--out_dir", str(out_dir),
    ]
    t0 = time.time()
    print(f"[RUN ] {time.strftime('%H:%M:%S')} {runner_module} -> {out_dir}",
          flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with log_path.open("w") as logf:
        print(f"Running command: {' '.join(cmd)}")
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
    elapsed = time.time() - t0
    if rc == 0:
        (out_dir / ".done").touch()
        print(f"[ OK ] {time.strftime('%H:%M:%S')} {runner_module} ({elapsed:.0f}s)",
              flush=True)
        return True
    print(f"[FAIL] {time.strftime('%H:%M:%S')} {out_dir} (see {log_path})",
          flush=True)
    return False


def out_root(scale: str, seed: int) -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "reproduce" / scale / f"seed_{seed}"
