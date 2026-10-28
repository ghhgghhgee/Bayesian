"""Display member baseline results without saving experiments or figures.

From the repository root:
python -m scripts.baselines_member_display
python -m scripts.baselines_member_display --config configs/baselines_member/paper.json
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from bcs.data import generate
from bcs.metrics import metrics


ROOT = Path(__file__).resolve().parents[1]


def evaluate(config, registry):
    """Compute only in memory using the framework's shared data and metrics."""
    rows, traces = [], {}
    for m, k in config["cases"]:
        successes = {name: 0 for name in config["algorithms"]}
        for trial in range(config["trials"]):
            A, y, truth = generate(config["n"], m, k, config["seed"], trial,
                                   config["amplitudes"])
            for name in config["algorithms"]:
                item = registry[name]
                module, function = item["entrypoint"].split(":")
                key = f"m{m}_k{k}_t{trial}_{name}"
                seed = int.from_bytes(hashlib.sha256(
                    f"{config['seed']}:{key}".encode()).digest()[:4], "little")
                result = getattr(importlib.import_module(module), function)(
                    A.copy(), y.copy(), seed=seed, max_iter=config["max_iter"],
                    tol=config["tol"], **item.get("params", {}))
                score = metrics(result.x, truth, config["success_threshold"])
                successes[name] += int(score["success"])
                if (m, k, trial) == (30, 3, 0):
                    db = [metrics(x, truth)["paper_mse_db"] for x in result.history]
                    traces[name] = (np.asarray(result.times), np.asarray(db))
        for name, count in successes.items():
            rows.append(dict(m=m, k=k, algorithm=name, trials=config["trials"],
                             successes=count, success_rate=count/config["trials"]))
            print(f"M={m:2d}, K={k:2d}, {name:14s}: {count}/{config['trials']} "
                  f"({100*count/config['trials']:.2f}%)", flush=True)
    return rows, traces


def display(config, rows, traces):
    """Create interactive figures only; never call savefig or export APIs."""
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, variable, fixed, value in ((axes[0], "m", "k", 3), (axes[1], "k", "m", 30)):
        for name in config["algorithms"]:
            selected = sorted((r for r in rows if r[fixed] == value and r["algorithm"] == name),
                              key=lambda r: r[variable])
            ax.plot([r[variable] for r in selected], [r["success_rate"] for r in selected],
                    "o-", markersize=4, label=name)
        ax.set(xlabel=variable.upper(), ylabel="Recovery success rate", ylim=(-.03, 1.03),
               title=f"{fixed.upper()}={value}; {config['trials']} trials per setting")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    if traces:
        _, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for name, (times, db) in traces.items():
            style = "o" if len(db) == 1 else "-"
            axes[0].plot(times, np.maximum(db, -160), style, label=name)
            axes[1].plot(np.arange(1, len(db)+1), np.maximum(db, -160), style, label=name)
        for ax in axes:
            ax.set(ylabel="Paper error [dB]", ylim=(-160, 5), title="M=30, K=3, trial=0")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
        axes[0].set_xlabel("Cumulative time [s]")
        axes[1].set_xlabel("Completed updates (BP: final estimate only)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/baselines_member/smoke.json")
    parser.add_argument("--registry", type=Path, default=ROOT/"algorithms_baselines_member.json")
    parser.add_argument("--no-show", action="store_true", help="Check plotting without opening a GUI")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    registry = json.loads(args.registry.read_text(encoding="utf-8-sig"))
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    rows, traces = evaluate(config, registry)
    display(config, rows, traces)
    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close("all")
    else:
        plt.show()
    print("Completed in memory; no experiment results or figures were saved.")


if __name__ == "__main__":
    main()
