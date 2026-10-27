#!/usr/bin/env python3
"""Run one inspected MoSca stage on private prepared copies, preserving predecessors."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['camera','scaffold','photo'],required=True)
    for name in ['mosca','workspace','output','config']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();os.environ['GS_BACKEND']='native_add3'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.custom_stages import validate_predecessor,validate_binding,require_dynamic_evidence
    from scripts.custom_tracking import digest
    validate_predecessor(a.output,a.stage)
    target={'camera':'bundle/bundle_cams.pth','scaffold':'mosca/mosca.pth','photo':'photometric_cam.pth'}[a.stage]
    if (a.output/target).exists():raise FileExistsError('stage output already exists; preserve it and choose a new copy')
    a.output.mkdir(parents=True,exist_ok=a.stage!='camera')
    sys.path.insert(0,str(a.mosca.resolve()))
    import torch,numpy as np
    from omegaconf import OmegaConf
    from lib_moca import camera as canonical_camera
    from scripts.custom_stages import install_camera_alias
    install_camera_alias(canonical_camera,sys.modules)
    from lite_moca_reconstruct import static_reconstruct
    from mosca_reconstruct import scaffold_reconstruct,photometric_reconstruct
    from lib_prior.prior_loading import Saved2D
    from lib_moca.epi_helpers import analyze_track_epi,identify_tracks
    cfg=OmegaConf.create(json.loads(a.config.read_text()))
    assert cfg.depth_dirname=='external_depth' and cfg.tap_mode=='bootstapir' and cfg.mode=='wild'
    original={str(p.relative_to(a.workspace)):digest(p) for p in a.workspace.rglob('*') if p.is_file()}
    validate_binding(a.output,a.stage,original)
    (a.output/(a.stage+'-config.json')).write_text(json.dumps(OmegaConf.to_container(cfg),indent=2))
    started=time.monotonic();report=dict(stage=a.stage,passed=False,input_sha256=original)
    try:
        if a.stage=='scaffold':
            # Match upstream loading before testing its unsafe recovery condition.
            s=(Saved2D(str(a.workspace)).load_dep(cfg.depth_dirname,cfg.depth_boundary_th)
                .normalize_depth(median_depth=cfg.dep_median).recompute_dep_mask(depth_boundary_th=cfg.depth_boundary_th)
                .load_track(cfg.tap_mode,min_valid_cnt=cfg.tap_loading_min_valid_cnt)
                .rescale_perframe_depth_from_bundle(bundle_pth_fn=str(a.output/'bundle/bundle.pth')).to('cuda:0'))
            with np.load(a.output/'tracker_epi.npz',allow_pickle=False) as epi:
                pairs=[tuple(x) for x in epi['continuous_pair_list'].tolist()]
                _,errors,_=analyze_track_epi(pairs,s.track,s.track_mask,s.H,s.W,epi['F_list'])
            static,dynamic=identify_tracks(errors,cfg.epi_th,static_cnt=1,dynamic_cnt=cfg.dyn_id_cnt)
            dynamic_np=dynamic.cpu().numpy()
            np.savez_compressed(a.output/'dynamic-evidence.npz',tracks=s.track.cpu().numpy(),
                visibility=s.track_mask.cpu().numpy(),dynamic=dynamic_np,static=static.cpu().numpy())
            report['dynamic_count_without_fallback']=require_dynamic_evidence(dynamic_np,getattr(cfg,'min_curve_num',32))
            del s;torch.cuda.empty_cache()
        function={'camera':static_reconstruct,'scaffold':scaffold_reconstruct,'photo':photometric_reconstruct}[a.stage]
        function(str(a.workspace),str(a.output),cfg)
        if not (a.output/target).is_file():raise RuntimeError('upstream omitted expected stage output')
        report['passed']=True
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            original_inputs_preserved=all(digest(a.workspace/n)==h for n,h in original.items()),
            output_sha256={str(p.relative_to(a.output)):digest(p) for p in a.output.rglob('*') if p.is_file()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not report['original_inputs_preserved']:report['passed']=False
        (a.output/(a.stage+'-report.json')).write_text(json.dumps(report,indent=2))
    if not report['original_inputs_preserved']:raise RuntimeError('input artifacts changed during stage')

if __name__=='__main__':main()
