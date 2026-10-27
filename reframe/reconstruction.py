"""Source-verified adapter for frozen MoSca checkpoints and calibrated rendering."""
import os
from pathlib import Path
import subprocess
import sys

from .media import sha256_file

MOSCA_REVISION='02404d2841b4e0dc2d9074860950bce5da579656'


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f'MoSca requires a CUDA GPU and its compiled rasterizer; detected torch {torch.__version__}, CUDA unavailable')


def nearby_camera(T_cw, distance):
    """Move camera along its local right axis, preserving orientation."""
    import torch
    T_wc=torch.linalg.inv(T_cw).clone()
    T_wc[:3,3]+=T_wc[:3,0]*distance
    return torch.linalg.inv(T_wc)


class FrozenMoSca:
    """Freeze both Gaussian pools; never save or modify original checkpoints."""
    def __init__(self, source_root, checkpoint_dir):
        require_cuda()
        import torch
        root=Path(source_root).resolve(); saved=Path(checkpoint_dir).resolve()
        revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if revision!=MOSCA_REVISION:
            raise ValueError('MoSca revision differs from the inspected adapter; review compatibility first')
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('tracked upstream modifications must be reviewed before using this pinned adapter')
        if 'lib_render.render_helper' in sys.modules:
            raise RuntimeError('MoSca renderer already imported; use a fresh process to select native_add3')
        os.environ['GS_BACKEND']='native_add3'
        sys.path.insert(0,str(root))
        from lib_moca.camera import MonocularCameras
        from lib_mosca.static_gs import StaticGaussian
        from lib_mosca.dynamic_gs import DynSCFGaussian
        from lib_render.render_helper import render
        names={'camera':'photometric_cam.pth','static':'photometric_s_model_native_add3.pth',
               'dynamic':'photometric_d_model_native_add3.pth'}
        self.fingerprints={k:sha256_file(saved/n) for k,n in names.items()}
        # weights_only prevents arbitrary pickle execution; never fall back to unsafe loading.
        states={k:torch.load(saved/n,map_location='cpu',weights_only=True) for k,n in names.items()}
        device=torch.device('cuda:0')
        self.camera=MonocularCameras.load_from_ckpt(states['camera']).to(device).eval()
        self.static=StaticGaussian.load_from_ckpt(states['static'],device=device).to(device).eval()
        self.dynamic=DynSCFGaussian.load_from_ckpt(states['dynamic'],device=device).to(device).eval()
        for model in [self.camera,self.static,self.dynamic]:
            model.requires_grad_(False)
        self.height=int(self.camera.default_H);self.width=int(self.camera.default_W)
        self.frame_count=self.camera.T
        if self.frame_count!=self.dynamic.T:
            raise ValueError('camera and scaffold frame counts differ')
        self.static_count=self.static.N
        self.count=self.static.N+self.dynamic.N
        self.device=device
        self.K=self.camera.K(self.height,self.width).detach()
        # List API handles delta cameras correctly at this pinned revision.
        self.poses=self.camera.T_cw_list().detach()
        self._render=render

    def gaussians(self,t):
        import torch
        if not isinstance(t,int) or not 0 <= t < self.frame_count:
            raise ValueError('time must be a reconstruction frame index')
        with torch.no_grad():
            static=self.static();dynamic=self.dynamic(t)
            result=tuple(torch.cat((s,d),dim=0).detach() for s,d in zip(static,dynamic))
        if len(result)!=5 or any(len(p)!=self.count or not torch.isfinite(p).all() for p in result):
            raise ValueError('invalid Gaussian parameters or row count')
        return result

    def render(self,gaussians,t,*,pose=None,colors=None):
        return self._render(gaussians,self.height,self.width,self.K,
                            self.poses[t] if pose is None else pose,bg_color=[0.,0.,0.],
                            colors_precomp=colors)

    def evidence(self,t,mask):
        from .association import extract_evidence
        import torch
        if tuple(mask.shape)!=(self.height,self.width):
            raise ValueError('mask and calibrated image dimensions differ')
        params=self.gaussians(t)
        m=torch.as_tensor(mask,device=self.device,dtype=torch.float32)
        e=extract_evidence(lambda c:self.render(params,t,colors=c)['rgb'],self.count,m)
        # Independent conservation against renderer's accumulated alpha image.
        with torch.no_grad():
            alpha=self.render(params,t)['alpha'].sum()
        if not torch.allclose(e[:,2].sum(),alpha,rtol=2e-3,atol=1e-2):
            raise ValueError('contribution sum differs from rendered alpha; backend is not validated')
        return e.cpu().numpy()

    def provenance(self):
        import torch
        return {'mosca_revision':MOSCA_REVISION,'backend':'native_add3',
                'checkpoint_sha256':self.fingerprints,'static_count':self.static_count,
                'dynamic_count':self.count-self.static_count,'frames':self.frame_count,
                'height':self.height,'width':self.width,'torch':torch.__version__,
                'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
