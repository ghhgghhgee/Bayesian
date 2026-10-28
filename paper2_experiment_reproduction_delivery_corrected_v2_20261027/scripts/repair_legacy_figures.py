"""Relabel archived exploration and restore raw dB; preserve original timing samples.

Requires the archived local fig4_histories_300_trials.npz (not distributed in Git).
This corrects presentation, not the original experimental protocol.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import reproduce_paper_figures as legacy

ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    run=ROOT/'results'; fig=run/'figures'; data=run/'data'
    old=rows(data/'fig4_trial0_curves.csv'); curves={}
    with np.load(data/'fig4_histories_300_trials.npz',allow_pickle=False) as saved:
        for name in legacy.FIG4_ALGORITHMS:
            selected=[r for r in old if r['algorithm']==name]
            history=saved[f't0_{name}_history']
            if len(history)!=len(selected):
                raise ValueError('Archived trajectory/timing length mismatch')
            curves[name]=dict(history=history,truth=saved[f't0_{name}_truth'],
                              times=np.array([float(r['elapsed_seconds_raw_median']) for r in selected]),
                              time_q25=np.array([float(r['elapsed_seconds_raw_q25']) for r in selected]),
                              time_q75=np.array([float(r['elapsed_seconds_raw_q75']) for r in selected]))
    legacy._figure4(curves,fig,data,'paper-aligned')
    for include,suffix in [(False,''),(True,'_with_focuss')]:
        legacy._plot_fig3(rows(data/'summary.csv'),fig,include,suffix)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_paper_figure_comparisons.py'),'--run',str(run)],check=True)
    path=run/'manifest.json'; manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['scope']='Archived exploratory reference implementation ONLY; not formal member results'
    manifest['limitations']=[
        'IRLS receives oracle true K; NP0 tol=1e-4 differs from the formal protocol.',
        'Artificial SBL endpoint alignment is NOT hardware calibration or evidence of timing validity.',
        'Original experiment predates source commit tracking. Correction hashes do not certify its historical source.',
        '2026-10-28 repair preserves measured timings, restores raw dB from saved trajectories, and relabels old figures.',
        'Formal member results live separately in results/team_members_20261028.']
    manifest['presentation_correction_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Archived figures corrected; original measured timings preserved.')

if __name__=='__main__':
    main()
