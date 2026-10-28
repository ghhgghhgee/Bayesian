"""Regression checks for PR4 protocol and data-display separation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('member5_plot',ROOT/'scripts/member5_plot_paper.py')
plot=importlib.util.module_from_spec(spec); spec.loader.exec_module(plot)

class Member5FigureTests(unittest.TestCase):
    def test_raw_db_is_not_clipped(self):
        raw,display,clipped=plot.db_values([[1+1e-9,0],[1,0],[2,0]],np.array([1.,0.]))
        self.assertLess(raw[0],-170); self.assertTrue(np.isneginf(raw[1]))
        np.testing.assert_equal(display,[-150,-150,0])
        np.testing.assert_equal(clipped,[True,True,False])

    def test_registry_call_has_no_truth_or_k_or_tol_override(self):
        cfg=plot.read_json(ROOT/'configs/team_members/paper.json')
        registry=plot.read_json(ROOT/'algorithms_baselines_member.json')
        for name in cfg['algorithms']:
            module,entry=registry[name]['entrypoint'].split(':')
            with patch(module+'.'+entry) as fn:
                plot.member_call(name,np.eye(2),np.ones(2),cfg,registry)
                args,kwargs=fn.call_args
                self.assertEqual(len(args),2)
                self.assertEqual(kwargs['tol'],1e-3)
                self.assertEqual(set(kwargs),{'seed','max_iter','tol'}|set(registry[name].get('params',{})))

    def test_irls_is_not_mislabeled_as_paper_irls0(self):
        self.assertEqual(plot.LABELS['IRLS_MEMBER'],'IRLS (p=0.5, member)')

    def test_formal_rejects_exploratory_config(self):
        with self.assertRaises(ValueError):
            plot.validate_run(ROOT,{'config':{'tol':1e-4}}, {})

if __name__=='__main__':
    unittest.main()
