"""Recompute saved metrics and paired data hashes; does not certify stopping rules."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcs.data import generate
from scripts._npz_arrays import NpzArrays


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def equal(a, b):
    return a == b or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    checks, failures = Counter(), Counter()
    examples = []

    def check(condition, name, context=""):
        checks[name] += 1
        if not condition:
            failures[name] += 1
            if len(examples) < 20:
                examples.append([name, str(context)])

    check(manifest.get("complete") and manifest.get("errors") == 0, "complete_without_errors")
    for name, expected in manifest["outputs"].items():
        p = directory / name
        check(p.is_file() and digest(p) == expected, "output_hash", name)
    trial_rows = rows(directory / "trials.csv")
    expected = {(m, k, t, a) for m, k in cfg["cases"] for t in range(cfg["trials"])
                for a in cfg["algorithms"]}
    by_key = {(int(r['m']), int(r['k']), int(r['trial']), r['algorithm']): r for r in trial_rows}
    check(len(by_key) == len(trial_rows), "unique_records")
    check(set(by_key) == expected, "complete_coverage")
    data = {}
    for m, k in cfg["cases"]:
        for trial in range(cfg["trials"]):
            A, y, truth = generate(cfg['n'], m, k, cfg['seed'], trial, cfg['amplitudes'])
            h = hashlib.sha256()
            for array in (A, y, truth):
                array = np.ascontiguousarray(array, dtype='<f8')
                h.update(str(array.shape).encode()); h.update(array.tobytes())
            data[m, k, trial] = A, y, truth, h.hexdigest()
    grouped = defaultdict(list)
    with NpzArrays(directory / 'estimates.npz') as estimates:
        check(set(estimates.files) == {f'm{m}_k{k}_t{t}_{a}' for m,k,t,a in expected}, 'estimate_coverage')
        for (m, k, trial, name), row in by_key.items():
            key = f'm{m}_k{k}_t{trial}_{name}'
            check(not row['error'], 'no_algorithm_error', key)
            if key not in estimates or (m, k, trial) not in data:
                check(False, 'missing_estimate_or_input', key)
                continue
            A, y, truth, h = data[m,k,trial]
            check(row['data_hash'] == h, 'regenerated_data_hash', key)
            x = estimates[key]
            check(x.shape == truth.shape and np.all(np.isfinite(x)), 'finite_estimate_shape', key)
            if x.shape != truth.shape or not np.all(np.isfinite(x)):
                continue
            err2 = float(np.dot(x-truth, x-truth))
            nmse = err2 / float(np.dot(truth, truth))
            relative = math.sqrt(nmse)
            mse = err2 / cfg['n']
            db = 10*math.log10(nmse) if nmse else -math.inf
            success = relative < cfg['success_threshold']
            for field, value in [('mse',mse),('nmse',nmse),('relative_error',relative),('paper_mse_db',db)]:
                check(equal(float(row[field]),value), 'metric_'+field, key)
            check((row['success'].lower()=='true') == success, 'success_label', key)
            check(np.linalg.norm(A@x-y)/np.linalg.norm(y) <= 1e-7, 'noiseless_feasibility', key)
            grouped[m,k,name].append((success,mse,nmse))
    summary = rows(directory / 'summary.csv')
    check(len(summary) == len(grouped), 'summary_row_count')
    seen = set()
    for row in summary:
        group = int(row['m']),int(row['k']),row['algorithm']
        check(group not in seen, 'unique_summary', group); seen.add(group)
        values = grouped[group]
        if not values:
            check(False,'missing_group',group); continue
        values = np.asarray(values)
        expected_stats = {'trials':len(values),'errors':0,'successes':sum(values[:,0]),
                          'success_rate':np.mean(values[:,0]),'mean_mse':np.mean(values[:,1]),
                          'mean_nmse':np.mean(values[:,2])}
        for field,value in expected_stats.items():
            check(equal(float(row[field]),float(value)), 'summary_'+field, group)
    return dict(directory=str(directory),passed=not failures,checks=dict(checks),
                failures=dict(failures),failure_examples=examples,manifest_sha256=digest(directory/'manifest.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+'); parser.add_argument('--out',required=True)
    args=parser.parse_args()
    runs=[audit(p) for p in args.directories]
    report=dict(passed=all(r['passed'] for r in runs),runs=runs,script_sha256=digest(__file__),
                scope='Direct metrics, success, summaries, input and output hashes. No bcs.metrics calls. Stopping rules and histories are covered by solver tests and run/compare, not this audit.')
    Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
