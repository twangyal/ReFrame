import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from test_selection import fixture
from reframe.selection import load_selection,revise_selection
from reframe.edit_render import render_edits


class EditRenderTests(unittest.TestCase):
    def test_full_timeline_same_rows_transformed_after_motion_and_hash_bound_outputs(self):
        class Scene:
            device='cpu';height=2;width=2;frame_count=5;static_count=2;count=4
            poses=torch.eye(4).repeat(5,1,1)
            def __init__(self,meta):self.meta=meta;self.calls=[]
            def provenance(self):return self.meta
            def gaussians(self,t):
                return (torch.tensor([[0.,0.,1.],[1.,0.,1.],[2.+t,0.,1.],[3.+t,0.,1.]]),
                        torch.eye(3).repeat(4,1,1),torch.ones(4,3),torch.ones(4,1),torch.zeros(4,3))
            def render(self,p,t,*,pose=None,colors=None):
                self.calls.append((t,tuple(x.clone() for x in p),pose.clone() if pose is not None else None,colors is not None))
                return {'rgb':colors.T.reshape(3,2,2) if colors is not None else torch.zeros(3,2,2)}
            def evidence(self,t,mask):return np.array([[1,0,1],[0,1,1],[1,0,1],[0,1,1]],float)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root);revise_selection(p,root/'candidate')
            s=load_selection(root/'candidate/render-selection.npz');scene=Scene(s['metadata'])
            recipe={'schema':1,'object_id':s['metadata']['object_id'],'selection_sha256':s['sha256'],
                    'color_rgb':[1.,.3,.1],'translation_extent_fraction':.25}
            masks=np.array([[[1,0],[1,0]]]*5,bool)
            r=render_edits(scene,s,masks,np.ones(5,bool),recipe,root/'out')
            self.assertEqual(len(r['files']),50)
            self.assertEqual(len(r['evaluation']),5)
            self.assertEqual(r['rendered_frames'],list(range(5)))
            self.assertFalse(r['phase_two_complete'])
            self.assertEqual(r['status'],'rendered_pending_visual_review')
            self.assertGreater(r['translation_world'][0],0)
            self.assertLess(r['maximum_translation_residual'],1e-5)
            self.assertEqual(r['evaluation'][0]['iou'],1.)
            self.assertEqual(r['evaluation'][0]['recolor_change']['outside_sam_absolute'],0.)
            # Five concrete variants in each view, then visible-mask render.
            first_calls=scene.calls[:11]
            self.assertEqual(len(first_calls),11)
            self.assertFalse(torch.equal(first_calls[0][1][4],first_calls[3][1][4]))
            self.assertFalse(torch.equal(first_calls[0][1][0],first_calls[4][1][0]))
            self.assertTrue(torch.equal(first_calls[0][1][0],first_calls[1][1][0]))
            self.assertTrue(torch.equal(first_calls[2][1][3][:,0],torch.tensor(s['selected'],dtype=torch.float32)))
            self.assertFalse(torch.equal(first_calls[0][2],first_calls[5][2]))
            for i in range(5):
                for a,b in zip(first_calls[i][1],first_calls[i+5][1]):self.assertTrue(torch.equal(a,b))
            self.assertTrue(first_calls[10][3])
            before=len(scene.calls)
            with self.assertRaises(ValueError):render_edits(scene,s,masks,np.ones(5,bool),{**recipe,'selection_sha256':'x'},root/'bad')
            self.assertEqual(before,len(scene.calls))
            with self.assertRaises(FileExistsError):render_edits(scene,s,masks,np.ones(5,bool),recipe,root/'out')
