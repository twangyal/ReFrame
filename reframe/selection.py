"""CPU revisions of frozen Gaussian membership; originals are never rewritten."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .association import aggregate_evidence
from .media import sha256_file

BINDING_KEYS=('checkpoint_sha256','mosca_revision','backend','static_count',
              'dynamic_count','frames','height','width')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _membership_digest(selected,static_count):
    return _digest({'static':np.flatnonzero(selected[:static_count]).tolist(),
                    'dynamic':np.flatnonzero(selected[static_count:]).tolist()})


def _indices(values,count,name):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or (a<0).any() or (a>=count).any() or len(np.unique(a))!=len(a):
        raise ValueError(f'invalid {name} row indices')
    return a.astype(np.int64)


def load_selection(path):
    """Read original/full or compact sidecar and validate pool-local row identity."""
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    meta=json.loads(str(data.pop('metadata')))
    for name in ('static_count','dynamic_count','frames','height','width'):
        if type(meta.get(name)) is not int or meta[name]<0 or (name not in ('static_count','dynamic_count') and meta[name]==0):
            raise ValueError(f'invalid selection {name}')
    n=meta['static_count']+meta['dynamic_count']
    if not n or not meta.get('object_id') or not meta.get('segmentation'):
        raise ValueError('missing selection identity')
    if set(meta.get('checkpoint_sha256',{}))!={'camera','static','dynamic'}:
        raise ValueError('missing checkpoint binding')
    for digest in meta['checkpoint_sha256'].values():
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid checkpoint digest')
    selected=np.zeros(n,dtype=bool)
    for pool,start in [('static',0),('dynamic',meta['static_count'])]:
        indices=_indices(data[pool+'_indices'],meta[pool+'_count'],pool)
        if meta.get('selected_'+pool+'_count')!=len(indices):raise ValueError('selection counts differ from indices')
        selected[start+indices]=True
    excluded=np.zeros(n,dtype=bool)
    for row in _validate_exclusions(meta.get('exclusions',[]),meta):
        excluded[row['index']+(meta['static_count'] if row['pool']=='dynamic' else 0)]=True
    if (selected & excluded).any():raise ValueError('an excluded row remains selected')
    train=meta['train_frames'];heldout=meta['heldout_frames']
    if (len(set(train))!=len(train) or len(set(heldout))!=len(heldout) or set(train)&set(heldout)
        or any(type(t) is not int or not 0<=t<meta['frames'] for t in train+heldout)):
        raise ValueError('invalid training/evaluation split')
    if 'revision_id' in meta:
        policy={k:meta.get(k) for k in ('parent_selection_sha256','aggregation','exclusions','membership_sha256')}
        if (policy['membership_sha256']!=_membership_digest(selected,meta['static_count'])
            or meta['revision_id']!=_digest(policy)):
            raise ValueError('selection revision or membership digest differs')
    elif 'state' not in data or 'evidence' not in data:
        raise ValueError('compact selection requires a verifiable revision')
    if 'state' in data:
        if data['state'].shape!=(n,) or not np.isin(data['state'],['foreground','background','unknown','ambiguous']).all():
            raise ValueError('invalid membership states')
        if not np.array_equal(selected,(data['state']=='foreground') & ~excluded):
            raise ValueError('selection indices differ from evidence states/exclusions')
    if 'evidence' in data:
        if data['evidence'].shape!=(len(meta['train_frames']),n,3):raise ValueError('training evidence shape differs')
    return {**data,'metadata':meta,'selected':selected,'sha256':sha256_file(path)}


def validate_scene_binding(selection,provenance):
    for key in BINDING_KEYS:
        if selection['metadata'].get(key)!=provenance.get(key):
            raise ValueError(f'selection belongs to another scene: {key}')


def _validate_exclusions(rows,meta):
    if not isinstance(rows,list):raise ValueError('exclusions must be a list')
    seen=set();result=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('invalid exclusion record')
        pool=row.get('pool');index=row.get('index');reason=row.get('reason')
        if (pool not in ('static','dynamic') or type(index) is not int
            or not 0<=index<meta[pool+'_count'] or not isinstance(reason,str) or not reason.strip()
            or (pool,index) in seen):raise ValueError('invalid, duplicate or unexplained exclusion')
        seen.add((pool,index));result.append({'pool':pool,'index':index,'reason':reason.strip()})
    return sorted(result,key=lambda r:(r['pool'],r['index']))


def revise_selection(path,out,*,threshold=None,min_frame_mass=None,exclusions=None):
    """Reaggregate saved training evidence and export a separate unrendered candidate.

    Exclusions are explicit manual ablations, never inferred from pool membership.
    An omitted policy keeps existing exclusions. New revisions carry no old IoU.
    """
    parent=load_selection(path);meta=parent['metadata']
    if 'evidence' not in parent:raise ValueError('full training evidence is required to revise membership')
    aggregation=dict(meta['aggregation'])
    if threshold is not None:aggregation['foreground_threshold']=threshold
    if min_frame_mass is not None:aggregation['min_frame_mass']=min_frame_mass
    rows=_validate_exclusions(meta.get('exclusions',[]) if exclusions is None else exclusions,meta)
    stats=aggregate_evidence(parent['evidence'],**aggregation)
    selected=stats['state']=='foreground'
    for row in rows:selected[row['index']+(meta['static_count'] if row['pool']=='dynamic' else 0)]=False
    policy={'parent_selection_sha256':parent['sha256'],'aggregation':aggregation,'exclusions':rows,
            'membership_sha256':_membership_digest(selected,meta['static_count'])}
    revision=_digest(policy)
    # Keep scene and segmentation provenance, not old runtime/quality claims.
    result_meta={k:meta[k] for k in (*BINDING_KEYS,'object_id','segmentation','train_frames','heldout_frames')}
    result_meta.update(policy,revision_id=revision,score_kind='visible foreground evidence fraction',
                       status='unrendered_exploratory_candidate',real_selection_success=False,
                       novel_view_review='pending rendering and visual inspection',
                       selected_static_count=int(selected[:meta['static_count']].sum()),
                       selected_dynamic_count=int(selected[meta['static_count']:].sum()))
    arrays={'static_indices':np.flatnonzero(selected[:meta['static_count']]),
            'dynamic_indices':np.flatnonzero(selected[meta['static_count']:]),'metadata':json.dumps(result_meta)}
    out=Path(out).resolve();source=Path(path).resolve()
    if out==source.parent or source.parent in out.parents:raise ValueError('revision outputs must be outside the original selection directory')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'selection.npz',**stats,evidence=parent['evidence'],**arrays)
    np.savez_compressed(out/'render-selection.npz',**arrays)
    training_mass=parent['evidence'][:,selected,:].sum(axis=(0,1),dtype=np.float64)
    report={**result_meta,'parent_selected_rows':int(parent['selected'].sum()),'selected_rows':int(selected.sum()),
            'added_rows':int((selected & ~parent['selected']).sum()),
            'removed_rows':int((parent['selected'] & ~selected).sum()),
            'training_selected_mass':dict(zip(['inside','outside','total'],training_mass.tolist())),
            'selection_sha256':sha256_file(out/'selection.npz'),
            'render_selection_sha256':sha256_file(out/'render-selection.npz'),
            'limitation':'Manual exclusions and tuning are exploratory; no rendered quality measured for this revision.'}
    (out/'revision.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
