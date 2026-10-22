"""Regenerate the three NP1 disagreement cases; no historical NPZ is required."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcs.data import generate, fingerprint
from bcs.metrics import metrics


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/np1_review_20261022')
    parser.add_argument('--historical-run', help='Optional historical run directory with estimates.npz')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    registry_path = ROOT/'algorithms.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8-sig'))
    arrays, summary, coordinates, trace, sources = {}, [], [], [], {}
    archived = None
    if args.historical_run:
        archived = np.load(Path(args.historical_run)/'estimates.npz', allow_pickle=False)
    try:
        for trial in (52,135,161):
            A,y,truth = generate(100,10,3,20261019,trial,'normal')
            prefix = f'trial{trial}'
            arrays.update({prefix+'_A':A, prefix+'_y':y, prefix+'_truth':truth})
            estimates = {}
            for name in ('NP1','NP1_MEMBER'):
                item = registry[name]
                module,function = item['entrypoint'].split(':')
                solver = getattr(importlib.import_module(module),function)
                source = Path(inspect.getsourcefile(solver))
                sources[name] = dict(item, source=str(source.relative_to(ROOT)),
                                     sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                key = f'm10_k3_t{trial}_{name}'
                seed = int.from_bytes(hashlib.sha256(f'20261019:{key}'.encode()).digest()[:4],'little')
                result = solver(A.copy(),y.copy(),seed=seed,max_iter=300,tol=1e-3,**item['params'])
                score = metrics(result.x,truth,1e-3)
                row = dict(trial=trial,algorithm=name,data_hash=fingerprint(A,y,truth),
                           algorithm_seed=seed,**score,iterations=len(result.history),
                           status=result.status,measurement_relative_residual=float(np.linalg.norm(A@result.x-y)/np.linalg.norm(y)))
                if archived is not None:
                    row['historical_match'] = bool(np.allclose(result.x,archived[key],rtol=1e-10,atol=1e-12))
                    if not row['historical_match']:
                        raise AssertionError('Fresh result differs from supplied historical run: '+key)
                summary.append(row)
                estimates[name] = result.x
                arrays[prefix+'_'+name+'_x'] = result.x
                arrays[prefix+'_'+name+'_history'] = np.asarray(result.history)
                for iteration,x in enumerate(result.history,1):
                    trace.append(dict(trial=trial,algorithm=name,iteration=iteration,
                                      elapsed_seconds=result.times[iteration-1],**metrics(x,truth,1e-3)))
            for index in range(100):
                coordinates.append(dict(trial=trial,index=index,truth=truth[index],
                                        reference=estimates['NP1'][index],member=estimates['NP1_MEMBER'][index],
                                        difference=estimates['NP1_MEMBER'][index]-estimates['NP1'][index]))
        np.savez_compressed(out/'cases.npz',**arrays)
        write_csv(out/'summary.csv',summary)
        write_csv(out/'coordinates.csv',coordinates)
        write_csv(out/'iterations.csv',trace)
        manifest = dict(purpose='NP1 reference/member discrepancy review; not algorithm correctness certification',
                        n=100,m=10,k=3,trials=[52,135,161],trial_indexing='zero-based',seed=20261019,
                        amplitudes='normal',max_iter=300,tol=1e-3,success_threshold=1e-3,
                        python=sys.version,numpy=np.__version__,platform=platform.platform(),algorithms=sources,
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        generator_sha256=hashlib.sha256((ROOT/'bcs/data.py').read_bytes()).hexdigest(),
                        registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                        historical_run=args.historical_run,
                        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        for row in summary:
            print(f"trial={row['trial']} {row['algorithm']} success={row['success']} relative_error={row['relative_error']:.9g}")
        print('Saved inputs, estimates, coordinate errors and full iteration metrics:',out)
    finally:
        if archived is not None:
            archived.close()


if __name__ == '__main__':
    main()
