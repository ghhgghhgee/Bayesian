"""CLI integration tests: errors and changed estimates must fail validation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True)

    def test_algorithm_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            p = Path(folder)
            config = dict(n=10, seed=3, trials=1, amplitudes='normal', max_iter=3,
                          tol=.001, success_threshold=.001, cases=[[5,1]], algorithms=['broken'])
            registry = {'broken': {'entrypoint': 'bcs.algorithms:solve', 'owner': 'test',
                                   'params': {'kind': 'INVALID'}}}
            (p/'config.json').write_text(json.dumps(config))
            (p/'registry.json').write_text(json.dumps(registry))
            run = subprocess.run([sys.executable, str(ROOT/'main.py'), 'run', '--config', str(p/'config.json'),
                                  '--registry', str(p/'registry.json'), '--out', str(p/'run'),
                                  '--data-dir', str(p/'data')], cwd=ROOT, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            manifest = json.loads((p/'run/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['errors'], 1)
            compare = subprocess.run([sys.executable, str(ROOT/'main.py'), 'compare', str(p/'run'), str(p/'run'),
                                      '--report', str(p/'compare.json')], cwd=ROOT, capture_output=True)
            self.assertNotEqual(compare.returncode, 0)

    def test_modified_estimate_fails_comparison(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            p = Path(folder)
            config = dict(n=10, seed=3, trials=1, amplitudes='normal', max_iter=3,
                          tol=.001, success_threshold=.001, cases=[[5,1]], algorithms=['NP0'])
            (p/'config.json').write_text(json.dumps(config))
            subprocess.run([sys.executable, str(ROOT/'main.py'), 'run', '--config', str(p/'config.json'),
                            '--out', str(p/'a'), '--data-dir', str(p/'data')], cwd=ROOT, check=True, capture_output=True)
            shutil.copytree(p/'a', p/'b')
            with np.load(p/'b/estimates.npz') as data:
                arrays = {key: data[key]+1 for key in data.files}
            np.savez_compressed(p/'b/estimates.npz', **arrays)
            run = subprocess.run([sys.executable, str(ROOT/'main.py'), 'compare', str(p/'a'), str(p/'b'),
                                  '--report', str(p/'compare.json')], cwd=ROOT, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            report = json.loads((p/'compare.json').read_text(encoding='utf-8'))
            self.assertFalse(report['checks']['estimates.npz'])


if __name__ == '__main__':
    unittest.main()
