"""Compare a leader replay with a member delivery without requiring member NPZ.

The member's published SHA256 of a deterministic estimates.npz can verify all
estimate arrays byte-for-byte. Wall-clock times are deliberately excluded.
"""
import os
for _var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_var]='1'

import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._npz_arrays import NpzArrays


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def compare_rows(left,right,ignored):
    counts=Counter(); examples=[]; total=0; max_difference=Counter()
    with Path(left).open(encoding='utf-8-sig',newline='') as a, Path(right).open(encoding='utf-8-sig',newline='') as b:
        for index,(la,rb) in enumerate(zip_longest(csv.DictReader(a),csv.DictReader(b))):
            total+=1
            if la is None or rb is None:
                counts['row_count']+=1
                if len(examples)<20:examples.append(dict(row=index,field='row_count'))
                continue
            fields=(set(la)|set(rb))-set(ignored)
            for field in fields:
                x,y=la.get(field),rb.get(field)
                if x==y:continue
                try:
                    xf,yf=float(x),float(y)
                    if math.isfinite(xf) and math.isfinite(yf):
                        max_difference[field]=max(max_difference[field],abs(xf-yf))
                    same=xf==yf or math.isclose(xf,yf,rel_tol=1e-10,abs_tol=1e-12)
                except (TypeError,ValueError):same=False
                if not same:
                    counts[field]+=1
                    if len(examples)<20:examples.append(dict(row=index,field=field,leader=x,member=y))
    return dict(passed=not counts,rows_compared=total,differences=dict(counts),examples=examples,
                max_absolute_numeric_differences=dict(max_difference))


def audit_histories(directory):
    directory=Path(directory)
    failures=Counter();checks=Counter();examples=[]
    def check(value,name,key):
        checks[name]+=1
        if not value:
            failures[name]+=1
            if len(examples)<20:examples.append([name,key])
    with (directory/'trials.csv').open(encoding='utf-8-sig',newline='') as stream, \
            NpzArrays(directory/'estimates.npz') as estimates, \
            NpzArrays(directory/'histories.npz') as histories:
        expected_estimates=set();expected_histories=set()
        for index,row in enumerate(csv.DictReader(stream),1):
            key=f"m{row['m']}_k{row['k']}_t{row['trial']}_{row['algorithm']}"
            expected_estimates.add(key);expected_histories.update([key,key+'_times'])
            if key not in estimates or key not in histories or key+'_times' not in histories:
                check(False,'missing_array',key);continue
            x=estimates[key];h=histories[key];t=histories[key+'_times']
            check(h.ndim==2 and h.shape==(int(row['iterations']),len(x)),'history_shape',key)
            check(np.all(np.isfinite(h)),'history_finite',key)
            check(t.shape==(len(h),) and np.all(np.isfinite(t)) and np.all(t>=0)
                  and np.all(np.diff(t)>=0),'timing_contract',key)
            check(len(h)>0 and np.array_equal(h[-1],x),'last_estimate_exact',key)
            if index%1800==0:print(f'History audit {index} arrays',flush=True)
        check(set(estimates.files)==expected_estimates,'complete_estimate_keys','all')
        check(set(histories.files)==expected_histories,'complete_history_keys','all')
    return dict(passed=not failures,checks=dict(checks),failures=dict(failures),examples=examples,
                histories_sha256=digest(directory/'histories.npz'),
                estimates_sha256=digest(directory/'estimates.npz'),
                note='Checks recorded trajectories; no member histories were available for cross-run trajectory comparison.')


def compare(leader,member,handoff,receipt,out,histories=False,history_report_path=None):
    leader,member=Path(leader),Path(member)
    lm,rm=read_json(leader/'manifest.json'),read_json(member/'manifest.json')
    def algorithms(m):
        return {key:{k:v for k,v in item.items() if k!='source_file'} for key,item in m['algorithms'].items()}
    computational=['main.py','bcs/algorithms.py','bcs/data.py','bcs/metrics.py','NP1.py',
                   'algorithms/np0.py','algorithms/baselines_member.py','algorithms_baselines_member.json',
                   'configs/team_members/paper.json']
    checks=dict(leader_complete_without_errors=lm.get('complete') is True and lm.get('errors')==0,
        member_complete_without_errors=rm.get('complete') is True and rm.get('errors')==0,
        same_config=lm['config']==rm['config'],same_algorithm_sources_and_params=algorithms(lm)==algorithms(rm),
        same_computational_sources=all(lm['source_hashes'].get(p)==rm['source_hashes'].get(p) for p in computational),
        same_dataset_sha256=lm['dataset_sha256']==rm['dataset_sha256'],
        same_environment=all(lm[k]==rm[k] for k in ['python','numpy','scipy','platform']),
        same_thread_limits=lm['thread_limits']==rm['thread_limits'])
    files=read_json(handoff)['files']
    expected=next(x['sha256'] for x in files if Path(x['path']).name=='estimates.npz')
    checks['member_estimates_hash_consistent']=expected==rm['outputs']['estimates.npz']
    actual=digest(leader/'estimates.npz')
    checks['all_final_estimates_byte_identical']=actual==expected
    receipt_data=read_json(receipt)
    checks['member_csv_download_verified']=receipt_data.get('verified') is True and receipt_data['sha256']==digest(member/'trials.csv')==rm['outputs']['trials.csv']
    trial_report=compare_rows(leader/'trials.csv',member/'trials.csv',{'seconds'})
    summary_report=compare_rows(leader/'summary.csv',member/'summary.csv',{'median_seconds'})
    expected_rows=len(lm['config']['cases'])*lm['config']['trials']*len(lm['config']['algorithms'])
    checks['complete_trial_comparison']=trial_report['passed'] and trial_report['rows_compared']==expected_rows
    checks['complete_summary_comparison']=summary_report['passed'] and summary_report['rows_compared']==len(lm['config']['cases'])*len(lm['config']['algorithms'])
    checks['leader_output_integrity']=all(digest(leader/p)==sha for p,sha in lm['outputs'].items())
    history_report=audit_histories(leader) if histories else None
    if history_report_path is not None:
        history_report=read_json(history_report_path)
        if (history_report['histories_sha256']!=lm['outputs']['histories.npz']
                or history_report['estimates_sha256']!=lm['outputs']['estimates.npz']):
            raise ValueError('History audit does not describe these archives')
    if history_report is not None:checks['leader_history_contract']=history_report['passed']
    report=dict(passed=all(checks.values()),checks=checks,trials=trial_report,summary=summary_report,
        final_estimates=dict(leader_sha256=actual,member_published_sha256=expected),
        leader_manifest_sha256=digest(leader/'manifest.json'),member_manifest_sha256=digest(member/'manifest.json'),
        histories=history_report,script_sha256=digest(__file__),
        source_scope=dict(all_snapshot_hashes_identical=lm['source_hashes']==rm['source_hashes'],
            reason='Leader adds a tested streaming scheduler. Only computational source hashes must match; documentation/source snapshot differences are retained, not hidden.'),
        limitations=['Wall-clock time is not compared.','Member histories and member final NPZ were not transferred; final arrays are verified using the published deterministic archive SHA256. No cross-run history equality is claimed.'])
    Path(out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=checks,out=str(out)),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leader',required=True);parser.add_argument('--member',required=True)
    parser.add_argument('--handoff',required=True);parser.add_argument('--receipt',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--histories',action='store_true')
    parser.add_argument('--history-report')
    args=parser.parse_args()
    result=compare(args.leader,args.member,args.handoff,args.receipt,args.out,args.histories,args.history_report)
    sys.exit(0 if result['passed'] else 1)
