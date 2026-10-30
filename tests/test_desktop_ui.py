import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from test_desktop_scene import bundle_fixture


@unittest.skipUnless(os.environ.get('DISPLAY'),'native GUI requires a display')
class DesktopUITests(unittest.TestCase):
    def test_real_widgets_selection_playback_session_and_export(self):
        import tkinter as tk
        from reframe.desktop import DesktopApp
        with tempfile.TemporaryDirectory() as d,patch('reframe.desktop.messagebox.showerror') as error:
            rootdir=Path(d);root=tk.Tk();app=DesktopApp(root,bundle_fixture(rootdir/'scene'))
            def pump(until=None,seconds=5):
                deadline=time.monotonic()+seconds
                while time.monotonic()<deadline:
                    root.update()
                    if until and until():return
                    time.sleep(.01)
                if until:self.fail('UI operation did not finish')
            try:
                pump(lambda:app.session is not None and not app.busy)
                root.update();self.assertEqual(str(app.mode_buttons['recolored'].cget('state')),'disabled')
                # Click the visible object through the actual canvas event mapping.
                x,y,w,h=app.rect
                app.canvas.event_generate('<Button-1>',x=int(x+6/16*w),y=int(y+5/12*h));root.update()
                self.assertTrue(app.session.selected)
                app.mode_buttons['recolored'].invoke();root.update();self.assertEqual(app.session.mode,'recolored')
                app.view_var.set('Nearby camera');app.view_combo.event_generate('<<ComboboxSelected>>');root.update()
                self.assertEqual(app.session.view,'nearby')
                # Native button Space activation must not also trigger the root shortcut.
                app.play_button.focus_force();root.update()
                app.play_button.event_generate('<KeyPress-space>');app.play_button.event_generate('<KeyRelease-space>');root.update()
                self.assertTrue(app.playing)
                app.stop_play();app.object_button.focus_force();root.update()
                selected=app.session.selected
                app.object_button.event_generate('<KeyPress-space>');app.object_button.event_generate('<KeyRelease-space>');root.update()
                self.assertNotEqual(app.session.selected,selected);self.assertFalse(app.playing)
                if not app.session.selected:app.object_button.invoke()
                app.mode_buttons['recolored'].invoke()
                old=app.session.frame;app.play_button.invoke();pump(seconds=.4);app.play_button.invoke()
                self.assertNotEqual(app.session.frame,old);self.assertFalse(app.playing)
                app.timeline.set(2);pump(seconds=.08);self.assertEqual(app.session.frame,2)
                session=rootdir/'chosen.reframe.json';app.save_session(session)
                app.reset_button.invoke();self.assertFalse(app.session.selected)
                app.load_session(session);pump(lambda:not app.busy)
                self.assertEqual((app.session.mode,app.session.frame),('recolored',2))
                output=rootdir/'chosen.mp4';app.start_export(output)
                self.assertTrue(app.busy);self.assertEqual(str(app.object_button.cget('state')),'disabled')
                pump(lambda:not app.busy,seconds=10)
                self.assertEqual(app.last_export,output);self.assertTrue(output.exists());error.assert_not_called()
            finally:
                if app.busy:pump(lambda:not app.busy,seconds=10)
                app.close()
