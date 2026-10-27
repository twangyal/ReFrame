import unittest
import numpy as np
from scripts.make_custom_preview import held_frame_indices,make_preview

class CustomPreviewTests(unittest.TestCase):
    def test_unequal_intervals_hold_original_samples_without_interpolation(self):
        indices=held_frame_indices([0.,2/24,5/24],6/24,24)
        np.testing.assert_array_equal(indices,[0,0,1,1,1,2])
        np.testing.assert_array_equal(held_frame_indices([0.,.083333,.166667],.25,24),[0,0,1,1,2,2])
        with self.assertRaises(ValueError):held_frame_indices([0,.2,.1],1,24)
        with self.assertRaises(ValueError):held_frame_indices([.1,.2],1,24)
        with self.assertRaises(ValueError):held_frame_indices([0,1],1,24)

    def test_encoded_preview_checks_hashes_and_preserves_timed_frame_count(self):
        import json,subprocess,tempfile,shutil
        from pathlib import Path
        from PIL import Image
        from reframe.media import sha256_file
        if not shutil.which('ffmpeg'):self.skipTest('ffmpeg required')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scene=root/'source';stages=root/'stages';priors=root/'priors';priors.mkdir()
            (scene/'images').mkdir(parents=True);(stages/'rgb-diagnostics').mkdir(parents=True)
            rows=[];inputs={};outputs={}
            for i,t in enumerate([0.,2/24,5/24]):
                name=f'{i:06d}.png';p=scene/'images'/name
                Image.new('RGB',(16,16),(i*80,20,40)).save(p)
                sha=sha256_file(p);inputs['images/'+name]=sha
                rows.append(dict(reconstruction_index=i,file=name,sha256=sha,source_pts_seconds=t))
                for view in ('source','nearby'):
                    key=f'rgb-diagnostics/{i:06d}_{view}.png';shutil.copyfile(p,stages/key);outputs[key]=sha
            (scene/'custom-scene.json').write_text(json.dumps(dict(frames=rows,width=16,height=16,source_duration_seconds=6/24)))
            (priors/'report.json').write_text(json.dumps(dict(prepared_manifest_sha256=sha256_file(scene/'custom-scene.json'))))
            inputs['report.json']=sha256_file(priors/'report.json')
            (stages/'photo-report.json').write_text(json.dumps(dict(passed=True,original_inputs_preserved=True,input_sha256=inputs,output_sha256=outputs)))
            video=make_preview(scene,stages,root/'preview',priors=priors)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(video)]))
            self.assertEqual(int(info['streams'][0]['nb_read_frames']),6)
            self.assertAlmostEqual(float(info['streams'][0]['duration']),.25,places=3)
            model_names=['photometric_cam.pth','photometric_s_model_native_add3.pth','photometric_d_model_native_add3.pth']
            model_hashes={}
            for name in model_names:
                (stages/name).write_bytes(b'checkpoint');model_hashes[name]=sha256_file(stages/name)
            (stages/'photo-report.json').write_text(json.dumps(dict(passed=True,original_inputs_preserved=True,input_sha256=inputs,output_sha256=model_hashes)))
            (stages/'playback-report.json').write_text(json.dumps(dict(stage='playback',passed=True,training_performed=False,
                original_trained_checkpoints_preserved=True,original_inputs_preserved=True,input_sha256=inputs,
                output_sha256={**outputs,**model_hashes,'photo-report.json':sha256_file(stages/'photo-report.json')})))
            make_preview(scene,stages,root/'reloaded-preview',priors=priors)
            (stages/model_names[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint binding'):make_preview(scene,stages,root/'changed-model',priors=priors)
            (stages/model_names[0]).write_bytes(b'checkpoint')
            (stages/'rgb-diagnostics/000000_source.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'render binding'):make_preview(scene,stages,root/'invalid',priors=priors)
            self.assertFalse((root/'invalid').exists())
            (scene/'custom-scene.json').write_text(json.dumps(dict(frames=rows,width=16,height=16,source_duration_seconds=.3)))
            with self.assertRaisesRegex(ValueError,'timing manifest binding'):make_preview(scene,stages,root/'changed-timing',priors=priors)
