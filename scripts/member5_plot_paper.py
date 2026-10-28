"""Plot the unified member run without fitting algorithms to published curves.

Run main.py first with configs/team_members/paper.json and the member registry.
This script reads its data, checks provenance, and times the fixed trial again.
"""
import os
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import k0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcs.data import generate

LABELS = {'SBL_MEMBER': 'SBL', 'NP1_MEMBER': 'NP-1', 'NP0_MEMBER': 'NP-0',
          'IRLS_MEMBER': 'IRLS (p=0.5, member)', 'FOCUSS_MEMBER': 'FOCUSS', 'BP_MEMBER': 'BP'}
STYLE = {'SBL_MEMBER': ('#c73a32', 'd'), 'NP1_MEMBER': ('#263f2d', 'o'),
         'NP0_MEMBER': ('#c98b24', 's'), 'IRLS_MEMBER': ('#4aafb3', '^'),
         'FOCUSS_MEMBER': ('#777777', 'x'), 'BP_MEMBER': ('#102ea8', '*')}
ORDER = ['SBL_MEMBER', 'NP1_MEMBER', 'NP0_MEMBER', 'IRLS_MEMBER', 'BP_MEMBER']
FIG4 = ['SBL_MEMBER', 'NP1_MEMBER', 'IRLS_MEMBER']
FLOOR = -150.0

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def db_values(history, truth):
    nmse = np.sum((np.asarray(history)-truth)**2, axis=1)/np.dot(truth, truth)
    with np.errstate(divide='ignore'):
        raw = 10*np.log10(nmse)
    return raw, np.maximum(raw, FLOOR), raw < FLOOR

def member_call(name, A, y, config, registry):
    # Ground truth / true K are intentionally absent from this interface.
    module, entry = registry[name]['entrypoint'].split(':')
    fn = getattr(importlib.import_module(module), entry)
    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:m30_k3_t0_{name}".encode()).digest()[:4], 'little')
    return fn(A.copy(), y.copy(), seed=seed, max_iter=config['max_iter'],
              tol=config['tol'], **registry[name].get('params', {}))

def validate_run(run, manifest, registry):
    cfg = manifest['config']
    if cfg != read_json(ROOT/'configs/team_members/paper.json'):
        raise ValueError('Formal figures require the unchanged member paper configuration')
    if not manifest.get('complete') or manifest.get('errors') != 0:
        raise ValueError('Incomplete or failed experiment')
    for name in cfg['algorithms']:
        item = manifest['algorithms'][name]
        if any(item.get(k) != v for k, v in registry[name].items()):
            raise ValueError(f'Registry mismatch: {name}')
        module, entry = registry[name]['entrypoint'].split(':')
        source = inspect.getsourcefile(getattr(importlib.import_module(module), entry))
        if digest(source) != item['sha256']:
            raise ValueError(f'Source mismatch: {name}')
    for name, expected in manifest['outputs'].items():
        if digest(run/name) != expected:
            raise ValueError(f'Output hash mismatch: {name}')

def save(fig, folder, name):
    fig.savefig(folder/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_success(rows, folder, all_six=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for panel, (ax, var, fixed, value) in enumerate(zip(axes, ['m', 'k'], ['k', 'm'], [3, 30])):
        order = ORDER if panel == 0 else ['SBL_MEMBER', 'BP_MEMBER', 'NP1_MEMBER', 'NP0_MEMBER', 'IRLS_MEMBER']
        if all_six:
            order = order + ['FOCUSS_MEMBER']
        for name in order:
            selected = sorted((r for r in rows if int(r[fixed]) == value and r['algorithm'] == name), key=lambda r: int(r[var]))
            color, marker = STYLE[name]
            ax.plot([int(r[var]) for r in selected], [float(r['success_rate']) for r in selected],
                    color=color, marker=marker, ms=4, lw=1, fillstyle='none', label=LABELS[name])
        ax.set(xlabel=var.upper(), ylabel='Success Rate', xlim=(0,30), ylim=(0,1.01),
               title=f"({'ab'[panel]}) {fixed.upper()}={value}")
        ax.set_xticks(range(0,31,5)); ax.set_yticks(np.linspace(0,1,11))
        ax.grid(ls=':', alpha=.6); ax.legend(fontsize=8, loc='upper right', fancybox=False)
    fig.suptitle('Six-member protocol; 300 trials/point; IRLS differs from paper IRLS0', fontsize=10)
    save(fig, folder, 'fig3_all_six' if all_six else 'fig3_success_rate')

def plot_cost(run, folder, cfg, registry):
    A, y, truth = generate(cfg['n'],30,3,cfg['seed'],0,cfg['amplitudes'])
    rows = []
    fig, axes = plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    with np.load(run/'histories.npz', allow_pickle=False) as saved:
        for name in FIG4:
            expected = saved[f'm30_k3_t0_{name}']
            samples = []
            for repeat in range(17):
                result = member_call(name,A,y,cfg,registry)
                if np.asarray(result.history).shape != expected.shape or not np.allclose(result.history,expected,rtol=1e-10,atol=1e-12):
                    raise ValueError(f'Timing replay differs from saved run: {name}')
                if repeat >= 2:
                    samples.append(result.times)
            med = np.median(samples,axis=0)
            q25,q75 = np.quantile(samples,[.25,.75],axis=0)
            raw, display, clipped = db_values(expected,truth)
            for i in range(len(raw)):
                rows.append(dict(algorithm=name,trial=0,m=30,k=3,iteration=i+1,benchmark_repeats=15,
                                 elapsed_seconds_raw_median=med[i],elapsed_seconds_raw_q25=q25[i],
                                 elapsed_seconds_raw_q75=q75[i],paper_mse_db=raw[i],raw_db=raw[i],
                                 display_db=display[i],display_clipped=bool(clipped[i])))
            color,_ = STYLE[name]
            marker = {'SBL_MEMBER':'s','NP1_MEMBER':'x','IRLS_MEMBER':'d'}[name]
            for ax,x in zip(axes,[med,np.arange(1,len(raw)+1)]):
                ax.plot(x,display,color=color,marker=marker,ms=3,lw=1,fillstyle='none',label=LABELS[name])
            axes[0].fill_betweenx(display,q25,q75,color=color,alpha=.15)
    for ax,label in zip(axes,['(a) Measured computation time (s)','(b) The number of iterations']):
        ax.set(xlabel=label,ylabel='MSE (dB)',ylim=(FLOOR,20),xlim=(0,None))
        ax.grid(ls=':',alpha=.6); ax.legend(fontsize=8,loc='upper right',fancybox=False)
    fig.suptitle('M=30, K=3, trial 0 (single realization); median of 15 serial timings\nDisplay floor: -150 dB; raw dB retained in CSV; shading: timing IQR',fontsize=10)
    save(fig,folder,'fig4_measured_trial0')
    write_csv(run/'fig4_trial0_curves.csv',rows)

def plot_density(run,folder):
    x=np.linspace(-5,5,301); x=x[x!=0]
    values=[k0(np.abs(x)/np.sqrt(2))/(np.pi*np.sqrt(2)),.5*np.exp(-np.abs(x)),np.exp(-x*x/4)/np.sqrt(4*np.pi)]
    fig,ax=plt.subplots(figsize=(6.3,4.3),layout='constrained')
    for value,label,style in zip(values,['Normal Product','Laplace','Normal'],['k-','r--','b-.']):
        ax.plot(x,value,style,label=label)
    ax.set(xlabel='x',ylabel='P(x)',xlim=(-5,5),ylim=(0,.9),title='Common variance = 2; NP density diverges at zero')
    ax.legend(); ax.grid(ls=':',alpha=.4)
    save(fig,folder,'fig1_pdf_comparison')
    write_csv(run/'fig1_pdf_values.csv',[dict(x=a,normal_product=b,laplace=c,normal=d) for a,b,c,d in zip(x,*values)])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--paper-crops',type=Path,required=True)
    args=parser.parse_args(); run=args.run.resolve()
    manifest=read_json(run/'manifest.json'); registry=read_json(ROOT/'algorithms_baselines_member.json')
    validate_run(run,manifest,registry)
    folder=run/'figures'; folder.mkdir(exist_ok=True)
    plot_density(run,folder)
    for all_six in [False,True]:
        plot_success(read_csv(run/'summary.csv'),folder,all_six)
    plot_cost(run,folder,manifest['config'],registry)
    for paper,name in [('fig1','fig1_pdf_comparison'),('fig3','fig3_success_rate'),('fig4','fig4_measured_trial0')]:
        fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
        for ax,path,label in zip(axes,[args.paper_crops/f'{paper}.png',folder/f'{name}.png'],['Published paper','Member implementation (protocol differences disclosed)']):
            ax.imshow(plt.imread(path)); ax.axis('off'); ax.set_title(label,fontsize=11)
        save(fig,folder,f'{paper}_paper_comparison')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    generated=[*folder.glob('*'),run/'fig1_pdf_values.csv',run/'fig4_trial0_curves.csv']
    plot_manifest=dict(experiment_manifest_sha256=digest(run/'manifest.json'),experiment_commit=manifest['git_commit'],
                       plotting_commit=commit,plotting_source_sha256=digest(__file__),
                       paper_crop_hashes={p.name:digest(p) for p in args.paper_crops.glob('*.png')},
                       outputs={str(p.relative_to(run)).replace('\\','/'):digest(p) for p in generated},
                       notes=['Measured serial timing only; no endpoint alignment.','Trial 0 fixed in advance, not selected for paper similarity.',
                              'IRLS_MEMBER p=0.5 is not the undocumented paper IRLS0.','FOCUSS included in data and six-curve supplement.'])
    (run/'plot_manifest.json').write_text(json.dumps(plot_manifest,indent=2),encoding='utf-8')
    print(f'Generated formal figures: {folder}')

if __name__=='__main__':
    main()
