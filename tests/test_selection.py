import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from reframe.association import aggregate_evidence
from reframe.media import sha256_file


def fixture(root):
    e=np.array([[[.9,.1,1],[.1,.9,1],[.85,.15,1],[.7,.3,1]]]*2)
    aggregation={'foreground_threshold':.8,'background_threshold':.2,'min_visible_frames':2,
                 'min_frame_mass':.1,'max_frame_weight':32.,'max_variance':.06}
    meta={'static_count':2,'dynamic_count':2,'frames':5,'height':2,'width':2,
          'object_id':'fixture-object','backend':'native_add3','mosca_revision':'fixture-revision',
          'checkpoint_sha256':{'camera':'a'*64,'static':'b'*64,'dynamic':'c'*64},
          'aggregation':aggregation,'train_frames':[1,3],'heldout_frames':[0,2,4],
          'segmentation':{'frame_manifest_sha256':'d'*64,'cache_key':'fixture-cache','sam_object_id':'0'},
          'selected_static_count':1,'selected_dynamic_count':1,
          'real_selection_success':True,'evaluation':[{'iou':1.0}]}
    (root/'original').mkdir()
    path=root/'original'/'selection.npz'
    np.savez_compressed(path,**aggregate_evidence(e,**aggregation),evidence=e,
                        static_indices=np.array([0]),dynamic_indices=np.array([0]),metadata=json.dumps(meta))
    return path


class SelectionTests(unittest.TestCase):
    def test_compact_policy_cannot_hide_excluded_rows_or_change_membership(self):
        from reframe.selection import revise_selection,load_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root)
            revise_selection(p,root/'candidate',exclusions=[{'pool':'static','index':0,'reason':'manual ablation'}])
            with np.load(root/'candidate/render-selection.npz',allow_pickle=False) as z:original=dict(z)
            for change in ('excluded-selected','overlapping-split','invalid-revision','same-count-substitution'):
                arrays=dict(original);meta=json.loads(str(arrays['metadata']))
                if change=='excluded-selected':
                    arrays['static_indices']=np.array([0]);meta['selected_static_count']=1
                elif change=='overlapping-split':meta['heldout_frames']=[0,1,4]
                elif change=='invalid-revision':meta['revision_id']='not-a-digest'
                else:arrays['dynamic_indices']=np.array([1])
                arrays['metadata']=json.dumps(meta);np.savez_compressed(root/'bad.npz',**arrays)
                with self.subTest(change=change),self.assertRaises(ValueError):load_selection(root/'bad.npz')

    def test_revision_excludes_only_explicit_row_without_reusing_old_quality(self):
        from reframe.selection import revise_selection,load_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root);before=sha256_file(p)
            report=revise_selection(p,root/'candidate',exclusions=[{'pool':'static','index':0,'reason':'manual outlier ablation'}])
            z=load_selection(root/'candidate'/'selection.npz')
            self.assertEqual(z['selected'].tolist(),[False,False,True,False])
            self.assertEqual(z['metadata']['object_id'],'fixture-object')
            self.assertEqual(z['metadata']['selected_static_count'],0)
            self.assertFalse(z['metadata']['real_selection_success'])
            self.assertNotIn('evaluation',z['metadata'])
            self.assertEqual(report['status'],'unrendered_exploratory_candidate')
            self.assertEqual(report['removed_rows'],1)
            self.assertEqual(sha256_file(p),before)
            np.testing.assert_allclose(z['score'],[.9,.1,.85,.7])
            compact=load_selection(root/'candidate'/'render-selection.npz')
            self.assertEqual(compact['selected'].tolist(),z['selected'].tolist())
            with self.assertRaises(FileExistsError):revise_selection(p,root/'candidate')

    def test_threshold_reaggregation_and_revision_id_are_reproducible(self):
        from reframe.selection import revise_selection,load_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root)
            a=revise_selection(p,root/'a',threshold=.9)
            b=revise_selection(p,root/'b',threshold=.9)
            self.assertEqual(a['revision_id'],b['revision_id'])
            self.assertEqual(load_selection(root/'a'/'selection.npz')['selected'].tolist(),[True,False,False,False])
            c=revise_selection(p,root/'c',threshold=.8)
            self.assertNotEqual(a['revision_id'],c['revision_id'])

    def test_invalid_exclusions_and_inconsistent_indices_are_rejected(self):
        from reframe.selection import revise_selection,load_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root)
            for rows in [[{'pool':'static','index':2,'reason':'out of range'}],
                         [{'pool':'dynamic','index':-1,'reason':'negative'}],
                         [{'pool':'dynamic','index':True,'reason':'boolean'}],
                         [{'pool':'dynamic','index':0,'reason':''}],
                         [{'pool':'dynamic','index':0,'reason':'x'}]*2]:
                with self.subTest(rows=rows),self.assertRaises(ValueError):
                    revise_selection(p,root/'bad',exclusions=rows)
                self.assertFalse((root/'bad').exists())
            with np.load(p,allow_pickle=False) as z:values=dict(z)
            values['dynamic_indices']=np.array([1])
            np.savez_compressed(root/'corrupt.npz',**values)
            with self.assertRaises(ValueError):load_selection(root/'corrupt.npz')

    def test_binding_rejects_different_checkpoint_or_row_counts(self):
        from reframe.selection import load_selection,validate_scene_binding
        with tempfile.TemporaryDirectory() as d:
            z=load_selection(fixture(Path(d)));meta=z['metadata']
            validate_scene_binding(z,meta)
            for key,value in [('checkpoint_sha256',{'camera':'f'*64}),('static_count',3),('backend','other')]:
                with self.subTest(key=key),self.assertRaises(ValueError):validate_scene_binding(z,{**meta,key:value})
