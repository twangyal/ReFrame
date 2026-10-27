"""The default/blocked runner must not require or contact Modal."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('modal_smoke', ROOT/'scripts/modal_smoke.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class ModalGateTests(unittest.TestCase):
    def test_default_only_prints_plan_without_sdk(self):
        r = subprocess.run([sys.executable, str(ROOT/'scripts/modal_smoke.py')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        plan = json.loads(r.stdout)
        self.assertFalse(plan['executed'])
        self.assertLess(plan['estimated_ceiling_usd'], 5)

    def test_missing_or_stale_evidence_refuses_launch(self):
        from datetime import datetime, timezone, timedelta
        base = dict(verified_at=datetime.now(timezone.utc).isoformat(),
                    workspace='test', balance_usd=30, no_overage_verified=True,
                    prior_trial_usage_usd=0,prior_billing_uncertainty_allowance_usd=.10,
                    rates_within_plan_verified=True, overhead_within_plan_verified=True,
                    secure_auth_verified=True, network_access_verified=True,
                    ephemeral_disk_cost_bound_verified=True,
                    no_concurrent_spend_verified=True, evidence='operator billing review')
        runner.validate_approval(base)
        for field in ('no_overage_verified','secure_auth_verified','network_access_verified',
                      'overhead_within_plan_verified','rates_within_plan_verified',
                      'no_concurrent_spend_verified','ephemeral_disk_cost_bound_verified'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_approval({**base, field:False})
        with self.assertRaises(ValueError):runner.validate_approval({**base, 'balance_usd':1})
        with self.assertRaises(ValueError):runner.validate_approval({**base, 'balance_usd':float('nan')})
        with self.assertRaises(ValueError):runner.validate_approval({**base, 'prior_trial_usage_usd':4.99})
        with self.assertRaises(ValueError):runner.validate_approval({**base, 'prior_billing_uncertainty_allowance_usd':float('nan')})
        with self.assertRaises(ValueError):runner.validate_approval({**base, 'verified_at':(datetime.now(timezone.utc)-timedelta(hours=3)).isoformat()})

    def test_execute_without_approval_stops_before_sdk(self):
        r = subprocess.run([sys.executable,str(ROOT/'scripts/modal_smoke.py'),'--execute'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('approval',r.stderr)
        self.assertNotIn('ModuleNotFoundError',r.stderr)

    def test_approval_cannot_launch_twice_even_if_key_order_changes(self):
        with tempfile.TemporaryDirectory() as d:
            runner.claim_approval({'workspace':'test','balance_usd':30},Path(d))
            with self.assertRaises(FileExistsError):
                runner.claim_approval({'balance_usd':30,'workspace':'test'},Path(d))

    def test_wrong_active_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.verify_workspace('different',{'workspace':'twangyal'})
        runner.verify_workspace('twangyal',{'workspace':'twangyal'})

    def test_explicit_gpu_guard_never_silently_accepts_a_different_allocation(self):
        from scripts.modal_smoke_worker import validate_gpu
        validate_gpu('NVIDIA A100-SXM4-40GB',40*1024**3,'A100-40GB')
        validate_gpu('NVIDIA A100-SXM4-80GB',80*1024**3,'A100-80GB')
        for name,size,expected in [('NVIDIA A100-SXM4-80GB',80*1024**3,'A100-40GB'),
                                   ('NVIDIA A100-SXM4-40GB',40*1024**3,'A100-80GB'),
                                   ('NVIDIA H100',80*1024**3,'A100-80GB'),
                                   ('NVIDIA A100',80*1024**3,'unknown')]:
            with self.subTest(name=name,expected=expected),self.assertRaises(ValueError):validate_gpu(name,size,expected)

    def test_authorized_a100_preference_accepts_only_40_or_80(self):
        from scripts.modal_smoke_worker import validate_gpu
        for size in (40,80):validate_gpu('NVIDIA A100',size*1024**3,'A100')
        for name,size in [('NVIDIA H100',80),('NVIDIA A100',20),('NVIDIA A100',60)]:
            with self.assertRaises(ValueError):validate_gpu(name,size*1024**3,'A100')
