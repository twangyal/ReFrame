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

    def test_fixed_positions_profile_requires_quality_photo_stage(self):
        for extra,ok in [(['--quality-fit','--fit-stage','photo'],True),([],False),(['--quality-fit','--fit-stage','camera'],False)]:
            result=subprocess.run([sys.executable,'scripts/modal_custom.py','--fixed-positions',*extra],capture_output=True,text=True)
            self.assertEqual(result.returncode==0,ok,result.stderr)
            if ok:
                plan=json.loads(result.stdout)
                self.assertTrue(plan['fixed_positions'])
                self.assertEqual(plan['input_timeout_seconds'],1500)
                self.assertLessEqual(plan['estimated_ceiling_usd'],4.30)

    def test_fixed_positions_config_rejects_other_experiment_changes(self):
        from scripts import modal_custom
        self.assertTrue(hasattr(modal_custom,'validate_quality_config'),'missing bounded experiment configuration validator')
        base=json.loads(Path('configs/custom-cat-quality-4000.json').read_text())
        fixed={**base,'photo_lr_np':0.0,'photo_lr_np_final':0.0}
        modal_custom.validate_quality_config(fixed,fixed_positions=True)
        for wrong in [base,{**fixed,'photo_lr_nq':0.0},{**fixed,'photo_total_steps':5000}]:
            with self.assertRaises(ValueError):modal_custom.validate_quality_config(wrong,fixed_positions=True)
        with self.assertRaises(ValueError):modal_custom.validate_quality_config(fixed)

    def test_frozen_scaffold_check_detects_position_or_identity_changes(self):
        import torch
        from scripts import custom_stages
        self.assertTrue(hasattr(custom_stages,'verify_fixed_positions'),'missing observed checkpoint invariant')
        original={'_node_xyz':torch.arange(18,dtype=torch.float32).reshape(2,3,3),'_node_certain':torch.ones(2,3,dtype=torch.bool),'_t_list':torch.arange(2)}
        fitted={'scf.'+k:v.clone() for k,v in original.items()}
        result=custom_stages.verify_fixed_positions(original,fitted)
        self.assertTrue(result['positions_unchanged'])
        for key in ['scf._node_xyz','scf._node_certain','scf._t_list']:
            changed={k:v.clone() for k,v in fitted.items()}
            changed[key].flatten()[0]=0 if key.endswith('certain') else 99
            with self.assertRaises(ValueError):custom_stages.verify_fixed_positions(original,changed)

    def test_frozen_position_config_cannot_bypass_named_experiment_guard(self):
        for flags in [[],['--quality-fit']]:
            result=subprocess.run([sys.executable,'scripts/modal_custom.py','--fit-stage','photo',
                '--config','configs/custom-cat-fixed-positions-4000.json',*flags],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,'generic path accepted fixed-position config without its guard')
            self.assertIn('fixed-positions',result.stderr)
