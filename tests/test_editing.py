import unittest
import numpy as np
import torch
from reframe.editing import edit_parameters, validate_edit_recipe, identity_summary


class EditingTests(unittest.TestCase):
    def params(self,t=0):
        return (torch.tensor([[0.,0.,2.],[1.+t,0.,3.],[2.+2*t,1.,4.]]),
                torch.eye(3).repeat(3,1,1),torch.ones(3,3),torch.tensor([[.4],[.7],[.9]]),
                torch.arange(36,dtype=torch.float32).reshape(3,12)/100)

    def test_translation_preserves_motion_and_unselected_rows_without_mutation(self):
        selected=np.array([True,False,True]);p0=self.params();p1=self.params(2)
        original=[x.clone() for x in p0]
        q0=edit_parameters(p0,selected,offset=[.2,-.3,.4])
        q1=edit_parameters(p1,selected,offset=[.2,-.3,.4])
        torch.testing.assert_close(q1[0]-q0[0],p1[0]-p0[0])
        torch.testing.assert_close(q0[0][selected]-p0[0][selected],torch.tensor([[.2,-.3,.4]]*2))
        torch.testing.assert_close(q0[0][1],p0[0][1])
        for before,after in zip(original,p0):self.assertTrue(torch.equal(before,after))
        for before,after in zip(p0[1:],q0[1:]):self.assertTrue(torch.equal(before,after))

    def test_recolor_resets_selected_higher_sh_only(self):
        p=self.params();q=edit_parameters(p,np.array([False,True,False]),color=[.2,.4,.8])
        torch.testing.assert_close(q[4][1,:3]*.28209479177387814+.5,torch.tensor([.2,.4,.8]))
        self.assertEqual(q[4][1,3:].count_nonzero(),0)
        self.assertTrue(torch.equal(q[4][[0,2]],p[4][[0,2]]))
        self.assertGreater(p[4][1,3:].count_nonzero(),0)
        for x,y in zip(p[:4],q[:4]):self.assertTrue(torch.equal(x,y))

    def test_invalid_inputs_fail(self):
        p=self.params()
        for selected in [np.array([1,0,1]),np.ones(2,bool),np.zeros(3,bool)]:
            with self.assertRaises(ValueError):edit_parameters(p,selected,color=[1,0,0])
        for kwargs in [dict(color=[1,0,float('nan')]),dict(color=[2,0,0]),dict(offset=[1,2]),dict(offset=[0,0,float('inf')])]:
            with self.assertRaises(ValueError):edit_parameters(p,np.ones(3,bool),**kwargs)

    def test_recipe_binds_exact_selection_and_object(self):
        s={'sha256':'a'*64,'metadata':{'object_id':'obj'}}
        r={'schema':1,'selection_sha256':'a'*64,'object_id':'obj','color_rgb':[.9,.2,.1],
           'translation_extent_fraction':.3}
        self.assertEqual(validate_edit_recipe(r,s),r)
        for change in [{'selection_sha256':'b'*64},{'object_id':'other'},{'translation_extent_fraction':float('nan')},{'unexpected':1}]:
            with self.assertRaises(ValueError):validate_edit_recipe({**r,**change},s)

    def test_identity_summary_keeps_pools_and_confidence_separate(self):
        s={'sha256':'a'*64,'selected':np.array([True,False,True]),'score':np.array([.9,.1,.8]),
           'variance':np.array([.01,.02,.03]),'visible_frames':np.array([2,1,3]),
           'metadata':{'object_id':'obj','static_count':2,'dynamic_count':1,'frames':3,
                       'checkpoint_sha256':{'camera':'x'},'aggregation':{'foreground_threshold':.8}}}
        out=identity_summary(s)
        self.assertEqual(out['selected_static'],1);self.assertEqual(out['selected_dynamic'],1)
        self.assertAlmostEqual(out['confidence']['minimum'],.8)
        self.assertFalse(out['geometric_stability_verified'])
        with self.assertRaises(ValueError):identity_summary({k:v for k,v in s.items() if k!='score'})
