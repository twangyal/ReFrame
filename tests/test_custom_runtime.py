import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


class CustomRuntimeTests(unittest.TestCase):
    def test_quality_plan_reserves_full_approved_runtime_cost(self):
        result=subprocess.run([sys.executable,'scripts/modal_custom.py','--quality-fit','--fit-stage','photo'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['input_timeout_seconds'],1500)
        self.assertEqual(plan['worker_timeout_seconds'],1400)
        self.assertEqual(plan['child_timeout_seconds'],1450)
        self.assertGreater(plan['estimated_ceiling_usd'],4.26)
        self.assertLessEqual(plan['estimated_ceiling_usd'],4.30)

    def test_quality_plan_cannot_be_used_for_unrelated_stage(self):
        result=subprocess.run([sys.executable,'scripts/modal_custom.py','--quality-fit','--fit-stage','camera'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)

    def test_extended_cost_is_used_by_cumulative_gate(self):
        from scripts.modal_smoke import validate_approval,PLAN
        evidence=dict(verified_at=datetime.now(timezone.utc).isoformat(),workspace='fixture',balance_usd=10,
            prior_trial_usage_usd=26,prior_billing_uncertainty_allowance_usd=.5,evidence='fixture')
        for key in ('no_overage_verified','rates_within_plan_verified','overhead_within_plan_verified','secure_auth_verified',
                    'network_access_verified','no_concurrent_spend_verified','ephemeral_disk_cost_bound_verified'):
            evidence[key]=True
        validate_approval(evidence,cumulative_budget_usd=30)
        with self.assertRaisesRegex(ValueError,'cumulative estimate'):
            validate_approval(evidence,cumulative_budget_usd=30,plan={**PLAN,'estimated_ceiling_usd':4.2672})

    def test_quality_config_changes_only_fitting_duration(self):
        base=json.loads(Path('configs/custom-cat-fit.json').read_text())
        candidate=json.loads(Path('configs/custom-cat-quality-4000.json').read_text())
        self.assertEqual(candidate,{**base,'photo_total_steps':4000})

    def test_nonfinite_loss_stops_before_backward_and_restores_method(self):
        import torch
        from scripts import custom_stages
        self.assertTrue(hasattr(custom_stages,'finite_loss_guard'),'missing nonfinite loss guard')
        original=torch.Tensor.backward
        with custom_stages.finite_loss_guard():
            x=torch.tensor(2.,requires_grad=True);(x*x).backward();self.assertEqual(x.grad.item(),4.)
            for value in [float('nan'),float('inf')]:
                x=torch.tensor(value,requires_grad=True)
                with self.assertRaisesRegex(FloatingPointError,'nonfinite'):x.backward()
                self.assertIsNone(x.grad)
        self.assertIs(torch.Tensor.backward,original)
