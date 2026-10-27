"""Run explicitly on CUDA with REFRAME_MOSCA_ROOT set; never a fake fallback."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from reframe.association import extract_evidence


@unittest.skipUnless(os.environ.get('REFRAME_MOSCA_ROOT'),'CUDA integration not requested; set REFRAME_MOSCA_ROOT on the GPU host')
class ActualRendererTests(unittest.TestCase):
    def test_gradient_matches_forward_basis_renders_with_occlusion(self):
        self.assertTrue(torch.cuda.is_available(),'CUDA required; this integration test cannot run on CPU')
        sys.path.insert(0,str(Path(os.environ['REFRAME_MOSCA_ROOT']).resolve()))
        os.environ['GS_BACKEND']='native_add3'
        from lib_render.render_helper import render
        d='cuda'
        mu=torch.tensor([[0.,0.,2.],[0.,0.,3.],[.2,0.,2.]],device=d)
        fr=torch.eye(3,device=d).repeat(3,1,1)
        scale=torch.tensor([[.2]*3,[.3]*3,[.1]*3],device=d)
        opacity=torch.tensor([[.9],[.9],[0.]],device=d)
        params=(mu,fr,scale,opacity,torch.zeros((3,3),device=d))
        K=torch.tensor([[32.,0,16.],[0,32.,16.],[0,0,1.]],device=d)
        pose=torch.eye(4,device=d)
        def rgb(c):return render(params,32,32,K,pose,bg_color=[0.,0.,0.],colors_precomp=c)['rgb']
        mask=torch.zeros((32,32),device=d);mask[:,:16]=1
        measured=extract_evidence(rgb,3,mask)
        # Each one-hot color render measures contribution directly in the forward
        # pass with ALL occluders retained. This does not reuse autograd evidence.
        reference=[]
        with torch.no_grad():
            for g in range(3):
                c=torch.zeros((3,3),device=d);c[g]=1
                w=rgb(c)[0]
                reference.append(torch.stack([(w*mask).sum(),(w*(1-mask)).sum(),w.sum()]))
        torch.testing.assert_close(measured,torch.stack(reference),rtol=2e-3,atol=2e-4)
        self.assertLess(measured[1,2],measured[0,2])
        torch.testing.assert_close(measured[2],torch.zeros(3,device=d))

if __name__=='__main__':unittest.main()
