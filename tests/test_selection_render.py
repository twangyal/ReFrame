import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from reframe.media import sha256_file
from test_selection import fixture


class SelectionRenderTests(unittest.TestCase):
    def test_absolute_mass_retained_for_small_contaminated_pool(self):
        from reframe.selection_render import selected_mass_metrics
        e=np.array([[0.,.1,.1],[0,0,0],[9.,1.,10.],[2.,0.,2.]])
        result=selected_mass_metrics(e,np.array([True,False,True,False]),2)
        self.assertAlmostEqual(result['static']['total'],.1)
        self.assertAlmostEqual(result['static']['outside_fraction'],1.)
        self.assertAlmostEqual(result['dynamic']['total'],10.)
        self.assertAlmostEqual(result['dynamic']['outside_fraction'],.1)
        empty=selected_mass_metrics(e,np.zeros(4,bool),2)
        self.assertIsNone(empty['static']['outside_fraction'])
        self.assertEqual(empty['dynamic']['total'],0.)
        with self.assertRaises(ValueError):selected_mass_metrics(e,np.ones(3,bool),2)

    def test_renderer_applies_sidecar_rows_and_keeps_occluders_for_visible_mask(self):
        import torch
        from reframe.selection import load_selection,revise_selection
        from reframe.selection_render import render_selection
        class ReferenceScene:
            # CPU fixture uses an explicit linear basis: not a MoSca quality test.
            device='cpu';height=2;width=2;frame_count=5;static_count=2;count=4
            poses=torch.eye(4).repeat(5,1,1)
            def __init__(self,meta):self.meta=meta
            def provenance(self):return self.meta
            def gaussians(self,t):
                return (torch.tensor([[0.,0.,1.]]*4),torch.eye(3).repeat(4,1,1),torch.ones(4,3),torch.ones(4,1),torch.zeros(4,3))
            def render(self,params,t,*,pose=None,colors=None):
                if colors is not None:
                    # Removing unselected opacity here would incorrectly erase occluders.
                    if not torch.equal(params[3],torch.ones(4,1)):raise AssertionError('occluders removed')
                    return {'rgb':colors.T.reshape(3,2,2)}
                return {'rgb':torch.zeros(3,2,2),'alpha':torch.ones(1,2,2)}
            def evidence(self,t,mask):return np.array([[1,0,1],[0,1,1],[1,0,1],[0,1,1]],dtype=float)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root);revise_selection(p,root/'candidate',exclusions=[{'pool':'static','index':0,'reason':'fixture'}])
            compact=root/'candidate/render-selection.npz';sel=load_selection(compact)
            masks=np.array([[[1,0],[1,0]]]*5,dtype=bool)
            report=render_selection(ReferenceScene(sel['metadata']),sel,masks,np.ones(5,bool),root/'render',times=[0,2,4])
            self.assertEqual(report['selection_sha256'],sha256_file(compact))
            self.assertAlmostEqual(report['evaluation'][0]['iou'],.5)
            self.assertEqual(report['evaluation'][0]['visible_mass']['static']['total'],0.)
            self.assertEqual(report['evaluation'][0]['visible_mass']['dynamic']['inside'],1.)
            self.assertEqual(report['evaluation'][0]['excluded_visible_mass']['static']['inside'],1.)
            self.assertEqual(report['evaluation'][0]['excluded_rows'][0]['index'],0)
            self.assertEqual(report['evaluation'][0]['excluded_rows'][0]['total'],1.)
            self.assertFalse(report['real_selection_success'])
            self.assertEqual(len(list((root/'render').glob('*_isolated.png'))),6)
            with self.assertRaises(FileExistsError):render_selection(ReferenceScene(sel['metadata']),sel,masks,np.ones(5,bool),root/'render',times=[0])
