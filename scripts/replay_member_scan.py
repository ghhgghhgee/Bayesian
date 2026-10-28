"""Replay an unchanged registry/config with bounded memory and case checkpoints.

Only scheduling/storage differ from main.py. Algorithm seeds, calls, record
fields and array ordering follow main.py. Timing is not a reproducibility key.
"""
from __future__ import annotations

import os
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as framework
from bcs.data import generate, fingerprint
from bcs.metrics import metrics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def npz_array(archive, key, value):
    # Same npy header, compression and zip64 settings as numpy.savez_compressed.
    with archive.open(key + '.npy', 'w', force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)


def provenance(config, registry):
    output = {}
    for name in config['algorithms']:
        module, function = registry[name]['entrypoint'].split(':')
        fn = getattr(importlib.import_module(module), function)
        path = inspect.getsourcefile(fn)
        output[name] = dict(registry[name], source_file=path, sha256=digest(path))
    return output


def run_case(task):
    cfg, registry, case, directory, signature = task
    m, k = case
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    completed = out/'checkpoint.json'
    if completed.exists():
        checkpoint = framework.read_json(completed)
        if checkpoint['signature'] != signature:
            raise ValueError('Checkpoint source/config mismatch')
        if all(digest(out/name) == value for name, value in checkpoint['outputs'].items()):
            return dict(case=case, directory=str(out), reused=True, errors=checkpoint['errors'])
        raise ValueError('Checkpoint output hash mismatch')
    functions = {}
    for name in cfg['algorithms']:
        module, function = registry[name]['entrypoint'].split(':')
        functions[name] = getattr(importlib.import_module(module), function)
    records = []
    errors = 0
    with zipfile.ZipFile(out/'estimates.npz', 'w', compression=zipfile.ZIP_DEFLATED) as estimates, \
            zipfile.ZipFile(out/'histories.npz', 'w', compression=zipfile.ZIP_DEFLATED) as histories:
        for trial in range(cfg['trials']):
            A, y, truth = generate(cfg['n'], m, k, cfg['seed'], trial, cfg['amplitudes'])
            data_hash = fingerprint(A, y, truth)
            for name in cfg['algorithms']:
                key = f'm{m}_k{k}_t{trial}_{name}'
                seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{key}".encode()).digest()[:4], 'little')
                row = dict(m=m, k=k, trial=trial, algorithm=name, data_hash=data_hash,
                           algorithm_seed=seed, success=False, mse='', nmse='', relative_error='',
                           paper_mse_db='', iterations=0, converged=False, status='error', seconds=0., error='')
                started = time.perf_counter()
                try:
                    result = functions[name](A.copy(), y.copy(), seed=seed,
                                             max_iter=cfg['max_iter'], tol=cfg['tol'],
                                             **registry[name].get('params', {}))
                    row['seconds'] = time.perf_counter()-started
                    row.update(metrics(result.x, truth, cfg['success_threshold']))
                    history = np.asarray(result.history, dtype=float)
                    times = np.asarray(result.times, dtype=float)
                    if (history.ndim != 2 or history.shape[1] != cfg['n'] or len(history) == 0
                            or len(times) != len(history) or not np.all(np.isfinite(history))
                            or not np.all(np.isfinite(times)) or np.any(times < 0)
                            or np.any(np.diff(times) < 0) or not np.array_equal(history[-1], result.x)):
                        raise ValueError('Invalid result history/times')
                    row.update(iterations=len(history), converged=bool(result.converged), status=result.status)
                    npz_array(estimates, key, np.asarray(result.x))
                    npz_array(histories, key, history)
                    npz_array(histories, key+'_times', times)
                except Exception as exc:
                    errors += 1
                    row.update(success=False, mse='', nmse='', relative_error='', paper_mse_db='',
                               status='error', error=f'{type(exc).__name__}: {exc}')
                records.append(row)
    framework.csv_write(out/'trials.csv', records)
    framework.write_json(completed, dict(signature=signature, case=case, errors=errors,
        outputs={name:digest(out/name) for name in ('trials.csv','estimates.npz','histories.npz')}))
    return dict(case=case, directory=str(out), reused=False, errors=errors)


def combine_archives(paths, target):
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            with zipfile.ZipFile(path) as source:
                for name in source.namelist():
                    with source.open(name) as src, output.open(name, 'w', force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, length=1024*1024)


def replay(config_path, registry_path, directory, data_dir, workers=2, resume=False):
    config = framework.read_json(config_path)
    registry = framework.read_json(registry_path)
    if workers < 1 or config['trials'] < 1 or config['max_iter'] < 1 or not config['cases']:
        raise ValueError('Invalid worker count or experiment size')
    if len(set(map(tuple,config['cases']))) != len(config['cases']):
        raise ValueError('Duplicate cases')
    out = Path(directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not resume:
        raise FileExistsError('Use a new output directory, or explicit --resume')
    algorithms = provenance(config, registry)
    signature = hashlib.sha256(json.dumps(dict(config=config,algorithms=algorithms),sort_keys=True).encode()).hexdigest()
    source_hashes = framework.source_hashes()
    if (out/'manifest.json').exists():
        old = framework.read_json(out/'manifest.json')
        if old.get('replay_signature') != signature or old['source_hashes'] != source_hashes:
            raise ValueError('Resume requires unchanged config, algorithm and source snapshot')
    dataset_path = framework.dataset(config, data_dir)
    def git(*args):
        p = subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    import platform
    import scipy
    manifest = dict(config=config,algorithms=algorithms,source_hashes=source_hashes,
        python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        git_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),
        dataset_sha256=digest(dataset_path),dataset_file=str(dataset_path),
        thread_limits={key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        complete=False,replay_signature=signature,
        scheduler=dict(entry='scripts/replay_member_scan.py',sha256=digest(__file__),workers=workers,
            note='Same solver calls, seeds and metrics as main.py; stream each case to disk. Timing differs.'))
    framework.write_json(out/'manifest.json',manifest)
    tasks = [(config,registry,case,str(out/'cases'/f'm{case[0]}_k{case[1]}'),signature) for case in config['cases']]
    # Workers write arrays themselves; only small metadata returns to the parent.
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_case,tasks,chunksize=1):
            results.append(result)
            print(f"[{len(results)}/{len(tasks)}] M={result['case'][0]}, K={result['case'][1]}: "
                  f"{config['trials']} trials, errors={result['errors']}, reused={result['reused']}",flush=True)
    records = []
    summaries = []
    for result in results:
        with (Path(result['directory'])/'trials.csv').open(encoding='utf-8',newline='') as stream:
            rows = list(csv.DictReader(stream))
        records.extend(rows)
        for name in config['algorithms']:
            selected = [r for r in rows if r['algorithm']==name]
            valid = [r for r in selected if r['status']!='error']
            successes = sum(r['success'].lower()=='true' for r in selected)
            summaries.append(dict(m=result['case'][0],k=result['case'][1],algorithm=name,
                trials=len(selected),errors=len(selected)-len(valid),successes=successes,
                success_rate=successes/len(selected),
                mean_mse=np.mean([float(r['mse']) for r in valid]) if valid else '',
                mean_nmse=np.mean([float(r['nmse']) for r in valid]) if valid else '',
                mean_iterations=np.mean([int(r['iterations']) for r in valid]) if valid else '',
                median_seconds=np.median([float(r['seconds']) for r in valid]) if valid else ''))
    framework.csv_write(out/'trials.csv',records)
    framework.csv_write(out/'summary.csv',summaries)
    for name in ('estimates.npz','histories.npz'):
        print('Assembling '+name,flush=True)
        combine_archives([Path(result['directory'])/name for result in results],out/name)
    errors = sum(r['errors'] for r in results)
    manifest.update(complete=True,errors=errors,outputs={p.name:digest(p) for p in out.iterdir()
                                                     if p.is_file() and p.name!='manifest.json'})
    framework.write_json(out/'manifest.json',manifest)
    print(f'Completed {len(records)} calls; errors={errors}',flush=True)
    if errors:
        raise RuntimeError('Algorithm errors recorded; inspect trials.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/team_members/paper.json')
    parser.add_argument('--registry',default='algorithms_baselines_member.json')
    parser.add_argument('--out',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    replay(args.config,args.registry,args.out,args.data_dir,args.workers,args.resume)


if __name__=='__main__':
    main()
