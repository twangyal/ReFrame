"""Minimal diagnostic image outputs; no web UI or editor."""
import numpy as np
from PIL import Image


def save_overlay(rgb,mask,path):
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if rgb.shape!=(*mask.shape,3) or mask.dtype!=bool:
        raise ValueError('overlay requires matching RGB and boolean mask')
    result=rgb.astype(float).copy()
    result[mask]=.55*result[mask]+.45*np.array([255.,40.,160.])
    Image.fromarray(np.clip(result,0,255).astype(np.uint8)).save(path)


def save_rgb(rgb,path):
    if hasattr(rgb,'detach'): rgb=rgb.detach().cpu().numpy()
    a=np.asarray(rgb)
    if a.shape[0]==3: a=np.moveaxis(a,0,-1)
    if not np.isfinite(a).all(): raise ValueError('nonfinite render')
    Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8)).save(path)


def selection_parameters(params,selected,mode):
    import torch
    selected=torch.as_tensor(selected,dtype=torch.bool,device=params[0].device)
    if selected.shape!=(len(params[0]),): raise ValueError('selection row count differs from scene')
    result=list(params)
    if mode=='isolated':
        result[3]=params[3]*selected[:,None]
    elif mode=='highlight':
        color=torch.tensor([1.,.1,.6],device=params[0].device)
        result[4]=params[4].clone()
        result[4][selected]=0
        result[4][selected,:3]=(color-.5)/.28209479177387814
    else:
        raise ValueError('mode must be highlight or isolated')
    return tuple(result)
