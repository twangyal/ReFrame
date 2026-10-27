import tempfile
import unittest
from pathlib import Path
from scripts import custom_stages

class CustomStageTests(unittest.TestCase):
    def test_stage_requires_completed_predecessor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(custom_stages.required_inputs('camera'),())
            for stage in ('scaffold','photo'):
                with self.assertRaises(FileNotFoundError):custom_stages.validate_predecessor(root,stage)
            for name in custom_stages.required_inputs('scaffold'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'checkpoint')
            custom_stages.validate_predecessor(root,'scaffold')
            with self.assertRaises(FileNotFoundError):custom_stages.validate_predecessor(root,'photo')
            with self.assertRaises(ValueError):custom_stages.required_inputs('invented')

    def test_insufficient_dynamic_evidence_refuses_upstream_static_fallback(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError,'dynamic'):
            custom_stages.require_dynamic_evidence(np.array([True]*31+[False]*40),minimum=32)
        self.assertEqual(custom_stages.require_dynamic_evidence(np.ones(32,dtype=bool)),32)

    def test_continuation_budget_cannot_exceed_existing_authorization(self):
        from datetime import datetime,timezone
        from scripts.modal_smoke import validate_approval
        evidence=dict(verified_at=datetime.now(timezone.utc).isoformat(),workspace='fixture',balance_usd=20,
            prior_trial_usage_usd=10,prior_billing_uncertainty_allowance_usd=.5,evidence='verified fixture')
        for key in ('no_overage_verified','rates_within_plan_verified','overhead_within_plan_verified','secure_auth_verified',
                    'network_access_verified','no_concurrent_spend_verified','ephemeral_disk_cost_bound_verified'):
            evidence[key]=True
        with self.assertRaises(ValueError):validate_approval(evidence)
        validate_approval(evidence,cumulative_budget_usd=30)
        with self.assertRaises(ValueError):validate_approval(evidence,cumulative_budget_usd=31)

    def test_camera_alias_is_explicit_and_refuses_conflicting_module(self):
        import types
        module=types.ModuleType('lib_moca.camera');registry={}
        custom_stages.install_camera_alias(module,registry)
        self.assertIs(registry['camera'],module)
        self.assertIs(registry['lib_mosca.camera'],module)
        custom_stages.install_camera_alias(module,registry)
        with self.assertRaisesRegex(ValueError,'conflicting'):
            custom_stages.install_camera_alias(types.ModuleType('different.camera'),registry)
        conflicting={'lib_mosca.camera':types.ModuleType('different.camera')}
        with self.assertRaisesRegex(ValueError,'conflicting'):
            custom_stages.install_camera_alias(module,conflicting)
        self.assertNotIn('camera',conflicting)

    def test_predecessor_must_pass_and_bind_same_inputs_and_checkpoint_bytes(self):
        import json,hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs={'images/000000.png':'framehash','external_depth/000000.npz':'depthhash','uniform_tap.npz':'trackhash'}
            outputs={}
            for name in custom_stages.required_inputs('scaffold'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'checkpoint')
                outputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            report=dict(passed=True,original_inputs_preserved=True,input_sha256=inputs,output_sha256=outputs)
            record=root/'camera-report.json';record.write_text(json.dumps(report))
            custom_stages.validate_binding(root,'scaffold',inputs)
            with self.assertRaises(ValueError):custom_stages.validate_binding(root,'scaffold',{**inputs,'uniform_tap.npz':'different'})
            report['passed']=False;record.write_text(json.dumps(report))
            with self.assertRaises(ValueError):custom_stages.validate_binding(root,'scaffold',inputs)
            report['passed']=True;record.write_text(json.dumps(report))
            (root/'bundle/bundle.pth').write_bytes(b'changed')
            with self.assertRaises(ValueError):custom_stages.validate_binding(root,'scaffold',inputs)

    def test_checkpoint_validation_rejects_nonfinite_and_empty_tensor_states(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pth'
            torch.save({'nested':{'positions':torch.ones(2,3)},'count':3},p)
            self.assertEqual(custom_stages.validate_checkpoint(p),6)
            torch.save({'positions':torch.tensor([float('nan')])},p)
            with self.assertRaisesRegex(ValueError,'nonfinite'):custom_stages.validate_checkpoint(p)
            torch.save({'metadata':'no model'},p)
            with self.assertRaisesRegex(ValueError,'empty'):custom_stages.validate_checkpoint(p)

    def test_curve_diagnostic_converts_normalized_rgb_for_pillow_without_mutation(self):
        import numpy as np
        from PIL import Image
        frame=np.array([[[0.,.5,1.],[-.1,1.1,.25]]],dtype=np.float32)
        original=frame.copy()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'diagnostic.png'
            with self.assertRaises(TypeError):Image.fromarray(frame).save(path)
            render=custom_stages.uint8_curve_renderer(lambda *args,**kwargs:[frame])
            result=render('camera',text='step',only_viz_last_frame=True)[0]
            Image.fromarray(result).save(path)
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved),[[[0,127,255],[0,255,63]]])
            np.testing.assert_array_equal(frame,original)
            self.assertEqual(result.dtype,np.uint8)
            bad=custom_stages.uint8_curve_renderer(lambda **kwargs:[np.full((2,2,3),np.nan)])
            with self.assertRaisesRegex(ValueError,'finite'):bad(only_viz_last_frame=True)

    def test_curve_video_retains_float_contract_for_upstream_scaling(self):
        import numpy as np
        frame=np.full((2,2,3),.5,dtype=np.float32)
        renderer=custom_stages.uint8_curve_renderer(lambda **kwargs:[frame])
        result=renderer(only_viz_last_frame=False)[0]
        self.assertIs(result,frame)
        np.testing.assert_array_equal((result*255).astype(np.uint8),np.full((2,2,3),127,dtype=np.uint8))

    def test_capture_retains_rgb_pixels_and_distinguishes_camera_views(self):
        import numpy as np
        from PIL import Image
        frame=np.zeros((8,12,3),dtype=np.uint8)
        frame[:4,4:8]=[12,34,56]
        class Camera:
            default_H=4
            default_W=4
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            wrapped=custom_stages.capture_rgb_diagnostics(lambda *a,**k:frame,root)
            self.assertIs(wrapped(3,None,Camera(),None),frame)
            wrapped(3,None,Camera(),None,T_cw='nearby',prefix_text='Novel')
            wrapped(3,None,Camera(),None,T_cw='fixed',prefix_text='Fix=0')
            self.assertEqual(sorted(p.name for p in root.iterdir()),['000003_nearby.png','000003_source.png'])
            with Image.open(root/'000003_source.png') as saved:
                np.testing.assert_array_equal(np.asarray(saved),np.broadcast_to([12,34,56],(4,4,3)))
