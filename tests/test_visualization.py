import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from PIL import Image

from reframe.visualization import selection_parameters, save_overlay


class VisualizationTests(unittest.TestCase):
    def test_isolation_preserves_row_order_and_original_parameters(self):
        p=(torch.tensor([[0.,0.,2.],[1.,0.,3.]]),torch.eye(3).repeat(2,1,1),
           torch.ones(2,3),torch.tensor([[.7],[.8]]),torch.zeros(2,12))
        q=selection_parameters(p,np.array([True,False]),'isolated')
        torch.testing.assert_close(q[3],torch.tensor([[.7],[0.]]))
        torch.testing.assert_close(p[3],torch.tensor([[.7],[.8]]))
        torch.testing.assert_close(q[0],p[0])
        h=selection_parameters(p,np.array([True,False]),'highlight')
        self.assertTrue(torch.equal(h[4][1],p[4][1]))
        self.assertTrue(torch.equal(p[4],torch.zeros(2,12)))
        self.assertFalse(torch.equal(h[4][0],p[4][0]))

    def test_overlay_modifies_only_masked_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'overlay.png'
            save_overlay(np.zeros((2,2,3),np.uint8),np.array([[1,0],[0,0]],bool),p)
            a=np.asarray(Image.open(p))
            self.assertGreater(a[0,0].sum(),0)
            self.assertEqual(a[1].sum(),0)
            self.assertEqual(a[0,1].sum(),0)

if __name__ == '__main__': unittest.main()
