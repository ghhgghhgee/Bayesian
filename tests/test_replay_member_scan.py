import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scan_replay',ROOT/'scripts/replay_member_scan.py')
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ScanReplayTests(unittest.TestCase):
    def test_streamed_npz_is_byte_identical_to_numpy(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            a,b=Path(folder)/'a.npz',Path(folder)/'b.npz'
            arrays={'x':np.arange(100,dtype=float).reshape(10,10),'y':np.zeros(3)}
            np.savez_compressed(a,**arrays)
            with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for key,value in arrays.items():
                    replay.npz_array(archive,key,value)
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_original_cli_and_streaming_replay_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            folder=Path(folder)
            cfg=dict(n=12,seed=20261019,trials=2,amplitudes='normal',max_iter=30,
                     tol=.001,success_threshold=.001,cases=[[7,1],[8,2]],
                     algorithms=['NP0_MEMBER','NP1_MEMBER','SBL_MEMBER','IRLS_MEMBER','FOCUSS_MEMBER','BP_MEMBER'])
            config=folder/'config.json';config.write_text(json.dumps(cfg),encoding='utf-8')
            registry=str(ROOT/'algorithms_baselines_member.json')
            commands=[['main.py','run','--config',str(config),'--registry',registry,
                       '--out',str(folder/'original'),'--data-dir',str(folder/'data_a')],
                      ['scripts/replay_member_scan.py','--config',str(config),'--registry',registry,
                       '--out',str(folder/'stream'),'--data-dir',str(folder/'data_b'),'--workers','2'],
                      ['main.py','compare',str(folder/'original'),str(folder/'stream'),
                       '--report',str(folder/'compare.json')]]
            for command in commands:
                result=subprocess.run([sys.executable,*command],cwd=ROOT,capture_output=True,text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(json.loads((folder/'compare.json').read_text())['passed'])
            self.assertEqual(replay.digest(folder/'original/estimates.npz'),replay.digest(folder/'stream/estimates.npz'))

    def test_resume_checks_source_and_output_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            cfg=dict(n=8,seed=2,trials=1,amplitudes='normal',max_iter=10,tol=.001,
                     success_threshold=.001,algorithms=['BP_MEMBER'])
            registry=json.loads((ROOT/'algorithms_baselines_member.json').read_text(encoding='utf-8'))
            task=(cfg,registry,[5,1],folder,'source-v1')
            self.assertFalse(replay.run_case(task)['reused'])
            self.assertTrue(replay.run_case(task)['reused'])
            with self.assertRaisesRegex(ValueError,'source/config'):
                replay.run_case((*task[:-1],'changed-source'))
            (Path(folder)/'trials.csv').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'output hash'):
                replay.run_case(task)


if __name__=='__main__':
    unittest.main()
