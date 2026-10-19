"""CLI: generate / run / compare / plot / package. See README.md."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import platform
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import scipy
from bcs.data import generate, fingerprint
from bcs.metrics import metrics

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for pattern in ("*.py", "*.json", "*.txt", "*.md", "*.ps1") for p in ROOT.rglob(pattern)
            if not any(part in (".git", "tmp", "data", "results", "dist", ".venv", "Projects")
                       for part in p.relative_to(ROOT).parts)}


def dataset(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = {k: config[k] for k in ("n", "seed", "trials", "amplitudes", "cases")}
    tag = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    path = directory / (tag + ".npz")
    if not path.exists():
        arrays = {}
        for m, k in config["cases"]:
            cases = [generate(config["n"], m, k, config["seed"], t, config["amplitudes"])
                     for t in range(config["trials"])]
            for index, name in enumerate(("A", "y", "x")):
                arrays[f"m{m}_k{k}_{name}"] = np.stack([c[index] for c in cases])
        np.savez_compressed(path, **arrays)
        write_json(path.with_suffix(".json"), signature)
    return path


def csv_write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    config = read_json(args.config)
    registry = read_json(args.registry)
    if config["trials"] < 1 or config["max_iter"] < 1 or not config["cases"]:
        raise ValueError("Empty experiment")
    if len(set(map(tuple, config["cases"]))) != len(config["cases"]):
        raise ValueError("Duplicate cases")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Output directory must be empty; use a new run name")
    functions, provenance = {}, {}
    for name in config["algorithms"]:
        item = registry[name]
        module, function = item["entrypoint"].split(":")
        functions[name] = getattr(importlib.import_module(module), function)
        file = inspect.getsourcefile(functions[name])
        provenance[name] = dict(item, source_file=file, sha256=digest(file))
    data_path = dataset(config, args.data_dir)
    data = np.load(data_path, allow_pickle=False)
    def git(*arguments):
        proc = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    manifest = dict(config=config, algorithms=provenance, source_hashes=source_hashes(),
                    python=sys.version, executable=sys.executable, numpy=np.__version__,
                    scipy=scipy.__version__, platform=platform.platform(),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
                    dataset_sha256=digest(data_path), dataset_file=str(data_path),
                    thread_limits={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}, complete=False)
    write_json(out / "manifest.json", manifest)
    records, estimates, histories, failures = [], {}, {}, 0
    with nullcontext():
        for m, k in config["cases"]:
            prefix = f"m{m}_k{k}"
            for trial in range(config["trials"]):
                A, y, truth = [data[f"{prefix}_{v}"][trial] for v in ("A", "y", "x")]
                data_hash = fingerprint(A, y, truth)
                for name in config["algorithms"]:
                    key = f"{prefix}_t{trial}_{name}"
                    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{key}".encode()).digest()[:4], "little")
                    record = dict(m=m, k=k, trial=trial, algorithm=name, data_hash=data_hash,
                                  algorithm_seed=seed, success=False, mse="", nmse="",
                                  relative_error="", paper_mse_db="", iterations=0,
                                  converged=False, status="error", seconds=0., error="")
                    start = time.perf_counter()
                    try:
                        result = functions[name](A.copy(), y.copy(), seed=seed,
                                                 max_iter=config["max_iter"], tol=config["tol"],
                                                 **registry[name].get("params", {}))
                        record["seconds"] = time.perf_counter()-start
                        record.update(metrics(result.x, truth, config["success_threshold"]))
                        history = np.asarray(result.history, dtype=float)
                        times = np.asarray(result.times, dtype=float)
                        if (history.ndim != 2 or history.shape[1] != config["n"] or
                            len(history) == 0 or len(times) != len(history) or
                            not np.all(np.isfinite(history)) or not np.all(np.isfinite(times)) or
                            np.any(times < 0) or np.any(np.diff(times) < 0) or
                            not np.allclose(history[-1], result.x, rtol=0, atol=0)):
                            raise ValueError("Invalid history/times or final estimate mismatch")
                        record.update(iterations=len(history), converged=bool(result.converged), status=result.status)
                        estimates[key] = np.asarray(result.x)
                        histories[key] = history
                        histories[key + "_times"] = times
                    except Exception as exc:
                        failures += 1
                        record.update(success=False, mse="", nmse="", relative_error="", paper_mse_db="",
                                      status="error", error=f"{type(exc).__name__}: {exc}")
                    records.append(record)
            csv_write(out / "trials.csv", records)
            print(f"Completed M={m}, K={k}: {config['trials']} trials, errors={failures}", flush=True)
    data.close()
    np.savez_compressed(out / "estimates.npz", **estimates)
    np.savez_compressed(out / "histories.npz", **histories)
    summaries = []
    for m, k in config["cases"]:
        for name in config["algorithms"]:
            rows = [r for r in records if (r["m"], r["k"], r["algorithm"]) == (m, k, name)]
            valid = [r for r in rows if r["status"] != "error"]
            summaries.append(dict(m=m, k=k, algorithm=name, trials=len(rows), errors=len(rows)-len(valid),
                                  successes=sum(r["success"] for r in rows),
                                  success_rate=sum(r["success"] for r in rows)/len(rows),
                                  mean_mse=np.mean([r["mse"] for r in valid]) if valid else "",
                                  mean_nmse=np.mean([r["nmse"] for r in valid]) if valid else "",
                                  mean_iterations=np.mean([r["iterations"] for r in valid]) if valid else "",
                                  median_seconds=np.median([r["seconds"] for r in valid]) if valid else ""))
    csv_write(out / "summary.csv", summaries)
    manifest.update(complete=True, errors=failures, outputs={p.name: digest(p) for p in out.iterdir() if p.name != "manifest.json"})
    write_json(out / "manifest.json", manifest)
    if failures:
        raise SystemExit(f"{failures} algorithm failures: inspect trials.csv")


def compare(left, right, report):
    left, right = Path(left), Path(right)
    lm, rm = read_json(left/"manifest.json"), read_json(right/"manifest.json")
    def algorithms(manifest):
        return {name: {k:v for k,v in item.items() if k != "source_file"}
                for name,item in manifest["algorithms"].items()}
    checks = {"completed_without_errors": all(m.get("complete") and m.get("errors") == 0 for m in (lm, rm)),
              "same_config": lm["config"] == rm["config"],
              "same_source": lm["source_hashes"] == rm["source_hashes"],
              "same_algorithms": algorithms(lm) == algorithms(rm),
              "same_environment": all(lm[k] == rm[k] for k in ("python", "numpy", "scipy", "platform"))}
    checks["output_integrity"] = all(
        (directory/name).is_file() and digest(directory/name) == expected
        for directory, manifest in ((left,lm), (right,rm))
        for name, expected in manifest.get("outputs", {}).items())
    def rows(path):
        with open(path/"trials.csv", encoding="utf-8") as stream:
            return [{k: v for k,v in row.items() if k != "seconds"} for row in csv.DictReader(stream)]
    checks["same_trials_metrics_and_data"] = rows(left) == rows(right)
    for filename in ("estimates.npz", "histories.npz"):
        with np.load(left/filename) as a, np.load(right/filename) as b:
            keys_a = {k for k in a.files if not k.endswith("_times")}
            keys_b = {k for k in b.files if not k.endswith("_times")}
            checks[filename] = keys_a == keys_b and all(np.allclose(a[k], b[k], rtol=1e-10, atol=1e-12) for k in keys_a)
    result = dict(passed=all(checks.values()), checks=checks, left=str(left), right=str(right),
                  note="Separate process replay; excludes wall-clock timing. Does not certify missing team submissions.")
    write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    config = manifest["config"]
    with open(directory/"summary.csv", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, variable, fixed, value in ((axes[0], "m", "k", 3), (axes[1], "k", "m", 30)):
        for name in config["algorithms"]:
            selected = sorted([r for r in rows if int(r[fixed]) == value and r["algorithm"] == name], key=lambda r: int(r[variable]))
            ax.plot([int(r[variable]) for r in selected], [float(r["success_rate"]) for r in selected],
                    "o-" if len(selected) > 2 else "o", label=name)
        ax.set(xlabel=variable.upper(), ylabel="Success rate", ylim=(-.03, 1.03), title=f"{fixed.upper()}={value}; {config['trials']} trials/point")
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.savefig(directory/"success_rate.png", dpi=180)
    plt.close(fig)
    if [30, 3] in config["cases"]:
        _, _, truth = generate(config["n"], 30, 3, config["seed"], 0, config["amplitudes"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        with np.load(directory/"histories.npz") as data:
            for name in ("NP1", "SBL", "IRLS"):
                key = f"m30_k3_t0_{name}"
                if key not in data:
                    continue
                db = [max(metrics(x, truth)["paper_mse_db"], -160) for x in data[key]]
                axes[0].plot(data[key+"_times"], db, ".-", label=name)
                axes[1].plot(np.arange(1,len(db)+1), db, ".-", label=name)
        for ax, label in zip(axes, ("Elapsed time (s)", "Iteration")):
            ax.set(xlabel=label, ylabel="Paper MSE (dB)", title="M=30, K=3; trial 0 (single realization)")
            ax.grid(alpha=.3)
            ax.legend()
        fig.savefig(directory/"convergence.png", dpi=180)
        plt.close(fig)
    manifest.setdefault("outputs", {}).update({p.name: digest(p) for p in directory.glob("*.png")})
    write_json(directory/"manifest.json", manifest)


def package(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    included = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(part in (".git", ".venv", "tmp", "dist", "__pycache__", "data", "Projects") for part in rel.parts):
                continue
            if p.suffix.lower() not in (".py", ".json", ".md", ".csv", ".png", ".npz", ".txt", ".ps1") and p.name not in (".gitignore", ".gitattributes"):
                continue
            if p.name.startswith("paper2_"):
                continue
            archive.write(p, str(rel))
            included.append(dict(path=str(rel), sha256=digest(p)))
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(included, indent=2, ensure_ascii=False))
    print(f"Packaged {len(included)} files -> {output}; sha256={digest(output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "run"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/smoke.json")
        p.add_argument("--data-dir", default="data")
        if command == "run":
            p.add_argument("--registry", default="algorithms.json")
            p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("left"); p.add_argument("right"); p.add_argument("--report", required=True)
    p = sub.add_parser("plot"); p.add_argument("directory")
    p = sub.add_parser("package"); p.add_argument("--out", default="dist/leader_submission.zip")
    args = parser.parse_args()
    if args.command == "generate":
        print(dataset(read_json(args.config), args.data_dir))
    elif args.command == "run":
        run(args)
    elif args.command == "compare":
        compare(args.left, args.right, args.report)
    elif args.command == "plot":
        plot(args.directory)
    elif args.command == "package":
        package(args.out)


if __name__ == "__main__":
    main()
