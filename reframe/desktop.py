"""Native offline desktop preview of verified Gaussian object edits.

Run: python -m reframe.desktop [--scene /path/to/scene.json]
No cloud SDK, credentials, model inference or network service is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk
from .desktop_scene import SceneBundle,MODES
from .desktop_session import DesktopSession,viewport_pixel
from .desktop_export import export_session

BG='#111722';PANEL='#1b2533';TEXT='#edf3fb';MUTED='#a5b4c8';ACCENT='#a5ef95'
MODE_LABELS={'baseline':'Original appearance','highlight':'Highlight selection',
             'isolated':'Isolate object','recolored':'Recolor · orange'}
VIEW_LABELS={'Original camera':'source','Nearby camera':'nearby'}


class DesktopApp:
    def __init__(self,root,scene_path=None):
        self.root=root;self.session=None;self.busy=False;self.playing=False;self.closed=False
        self.events=queue.Queue();self.executor=ThreadPoolExecutor(max_workers=1)
        self.photo=None;self.rect=(0,0,0,0);self.last_export=None;self.last_error=None
        self.play_timer=None;self.poll_timer=None;self.refresh_timer=None
        root.title('ReFrame — Gaussian object study');root.geometry('1220x820');root.minsize(960,700);root.configure(bg=BG)
        self._style();self._layout();self._bindings();self._controls();self._poll()
        root.protocol('WM_DELETE_WINDOW',self.close)
        if scene_path:self.load_scene(scene_path)

    def _style(self):
        style=ttk.Style(self.root);style.theme_use('clam')
        style.configure('.',font=('Helvetica',10),background=BG,foreground=TEXT)
        style.configure('TButton',background=PANEL,foreground=TEXT,padding=(11,8),borderwidth=0)
        style.map('TButton',background=[('active','#304359'),('disabled','#19212c')],foreground=[('disabled','#647388')])
        style.configure('Accent.TButton',background=ACCENT,foreground='#17251c')
        style.map('Accent.TButton',background=[('active','#c2ffb5'),('disabled','#304839')])
        style.configure('TCombobox',fieldbackground=PANEL,background=PANEL,foreground=TEXT,arrowcolor=TEXT,padding=6)
        style.map('TCombobox',fieldbackground=[('readonly',PANEL)],selectbackground=[('readonly',PANEL)],selectforeground=[('readonly',TEXT)])
        style.configure('TProgressbar',troughcolor=PANEL,background=ACCENT,borderwidth=0)

    def _label(self,parent,text='',size=10,color=TEXT,**kwargs):
        return tk.Label(parent,text=text,bg=parent.cget('bg'),fg=color,font=('Helvetica',size),**kwargs)

    def _layout(self):
        header=tk.Frame(self.root,bg=BG,padx=22,pady=17);header.pack(fill='x')
        self._label(header,'ReFrame',24,anchor='w').pack(side='left')
        self._label(header,'  /  OBJECT STUDY',10,MUTED).pack(side='left',padx=12)
        self.export_button=ttk.Button(header,text='Export MP4',style='Accent.TButton',command=self.choose_export);self.export_button.pack(side='right',padx=(8,0))
        self.save_button=ttk.Button(header,text='Save session',command=self.choose_save);self.save_button.pack(side='right',padx=4)
        self.open_session_button=ttk.Button(header,text='Open session',command=self.choose_session);self.open_session_button.pack(side='right',padx=4)
        self.open_button=ttk.Button(header,text='Open scene',command=self.choose_scene);self.open_button.pack(side='right',padx=4)
        body=tk.Frame(self.root,bg=BG);body.pack(fill='both',expand=True,padx=22)
        sidebar=tk.Frame(body,bg=PANEL,width=245,padx=16,pady=18);sidebar.pack(side='left',fill='y',padx=(0,16));sidebar.pack_propagate(False)
        self._label(sidebar,'SCENE OBJECT',10,MUTED,anchor='w').pack(fill='x')
        self.object_button=ttk.Button(sidebar,text='No scene loaded',command=self.toggle_selection);self.object_button.pack(fill='x',pady=(12,8))
        self.object_detail=self._label(sidebar,'Open a verified scene bundle',9,MUTED,justify='left',anchor='w',wraplength=210);self.object_detail.pack(fill='x')
        self._label(sidebar,'APPEARANCE',10,MUTED,anchor='w').pack(fill='x',pady=(28,8))
        self.mode_var=tk.StringVar(value='baseline');self.mode_buttons={}
        for mode,label in MODE_LABELS.items():
            button=tk.Radiobutton(sidebar,text=label,value=mode,variable=self.mode_var,command=self.change_mode,
              bg=PANEL,fg=TEXT,selectcolor='#304359',activebackground=PANEL,activeforeground=ACCENT,
              disabledforeground='#6c7a8d',font=('Helvetica',10),anchor='w',pady=8,highlightthickness=0)
            button.pack(fill='x');self.mode_buttons[mode]=button
        self._label(sidebar,'Orange is the verified color preset.',9,MUTED,wraplength=210,justify='left').pack(fill='x',pady=(6,15))
        self.undo_button=ttk.Button(sidebar,text='Undo appearance',command=self.undo);self.undo_button.pack(fill='x',pady=4)
        self.reset_button=ttk.Button(sidebar,text='Reset selection',command=self.reset);self.reset_button.pack(fill='x',pady=4)
        self._label(sidebar,'Movement unavailable',10,'#ffc487',anchor='w').pack(fill='x',pady=(25,5))
        self._label(sidebar,'Clean translation did not pass visual review.',9,MUTED,wraplength=210,justify='left',anchor='w').pack(fill='x')
        content=tk.Frame(body,bg=BG);content.pack(side='left',fill='both',expand=True)
        toolbar=tk.Frame(content,bg=BG);toolbar.pack(fill='x',pady=(0,10))
        self.scene_label=self._label(toolbar,'Open a scene to begin',13,anchor='w');self.scene_label.pack(side='left')
        self.view_var=tk.StringVar(value='Original camera');self.view_combo=ttk.Combobox(toolbar,textvariable=self.view_var,values=list(VIEW_LABELS),state='readonly',width=17)
        self.view_combo.pack(side='right');self.view_combo.bind('<<ComboboxSelected>>',self.change_view)
        self.canvas=tk.Canvas(content,bg='#090e15',highlightthickness=1,highlightbackground='#304053')
        self.canvas.pack(fill='both',expand=True);self.canvas.bind('<Configure>',self.resize);self.canvas.bind('<Button-1>',self.viewport_click)
        details=tk.Frame(content,bg=BG);details.pack(fill='x',pady=(9,4))
        self.selection_label=self._label(details,'Click the balloon or select it in the object list.',9,MUTED,anchor='w');self.selection_label.pack(side='left')
        self.frame_label=self._label(details,'—',9,MUTED);self.frame_label.pack(side='right')
        transport=tk.Frame(content,bg=BG);transport.pack(fill='x',pady=(4,10))
        self.previous_button=ttk.Button(transport,text='<',width=3,command=lambda:self.step(-1));self.previous_button.pack(side='left')
        self.play_button=ttk.Button(transport,text='Play',width=7,command=self.toggle_play);self.play_button.pack(side='left',padx=6)
        self.next_button=ttk.Button(transport,text='>',width=3,command=lambda:self.step(1));self.next_button.pack(side='left')
        self.timeline=tk.Scale(transport,from_=0,to=0,orient='horizontal',showvalue=False,command=self.scrub,
          bg=BG,fg=TEXT,troughcolor=PANEL,activebackground=ACCENT,highlightthickness=0,bd=0,sliderlength=18)
        self.timeline.pack(side='left',fill='x',expand=True,padx=14)
        self._label(content,'Saved GPU renders  ·  Discrete states at chosen playback speed  ·  No live reconstruction',9,MUTED,anchor='w').pack(fill='x',pady=(0,10))
        footer=tk.Frame(self.root,bg=BG,padx=22,pady=12);footer.pack(fill='x')
        self.status=tk.StringVar(value='Ready. Open a local scene.json bundle; no network or GPU is used.')
        self.status_label=self._label(footer,'',9,MUTED,anchor='w');self.status_label.configure(textvariable=self.status);self.status_label.pack(fill='x')
        self.progress=ttk.Progressbar(footer,mode='indeterminate');self.progress.pack(fill='x',pady=(8,0))

    def _bindings(self):
        self.root.bind('<space>',lambda e:self.transport_key(e,self.toggle_play))
        self.root.bind('<Left>',lambda e:self.transport_key(e,lambda:self.step(-1)));self.root.bind('<Right>',lambda e:self.transport_key(e,lambda:self.step(1)))
        self.root.bind('<Control-z>',lambda e:self.undo());self.root.bind('<Control-o>',lambda e:self.choose_scene())
        self.root.bind('<Control-s>',lambda e:self.choose_save());self.root.bind('<Control-e>',lambda e:self.choose_export())

    def transport_key(self,event,action):
        # Buttons/scales/comboboxes retain their own native keyboard behavior.
        # Root bindtags run after widget-class bindings: do not activate twice.
        if event.widget in (self.root,self.canvas):
            action();return 'break'

    def _controls(self):
        ready=self.session is not None and not self.busy
        for button in [self.open_button,self.open_session_button]:button.configure(state='disabled' if self.busy else 'normal')
        for button in [self.save_button,self.export_button,self.object_button,self.play_button,self.previous_button,self.next_button,self.reset_button]:
            button.configure(state='normal' if ready else 'disabled')
        self.undo_button.configure(state='normal' if ready and self.session.history else 'disabled')
        self.timeline.configure(state='normal' if ready else 'disabled');self.view_combo.configure(state='readonly' if ready else 'disabled')
        for mode,button in self.mode_buttons.items():
            button.configure(state='normal' if ready and (mode=='baseline' or self.session.selected) else 'disabled')

    def _error(self,error):
        self.last_error=str(error);self.status.set('Error: '+str(error));messagebox.showerror('ReFrame',str(error),parent=self.root)

    def _job(self,function,done,message):
        if self.busy:return
        self.stop_play();self.busy=True;self._controls();self.status.set(message);self.progress.start(12)
        future=self.executor.submit(function)
        future.add_done_callback(lambda f:self.events.put(('done',f,done)))

    def _poll(self):
        if self.closed:return
        try:
            while True:
                kind,*args=self.events.get_nowait()
                if kind=='progress':self.status.set(f'Preparing export frames {args[0]}/{args[1]}… encoding and verification follow.')
                else:
                    future,done=args;self.busy=False;self.progress.stop()
                    try:done(future.result())
                    except Exception as error:self._error(error)
                    self._controls()
        except queue.Empty:pass
        self.poll_timer=self.root.after(50,self._poll)

    def _adopt(self,session):
        self.session=session;self.timeline.configure(to=session.scene.frames-1)
        self.scene_label.configure(text=session.scene.data['title']+'  /  saved scene')
        o=session.scene.data['object'];total=o['dynamic_count']+o['static_count']
        self.object_detail.configure(text=f'{total:,} Gaussians\nPersistent ID: {o["id"][:8]}...\nMembership threshold: {o.get("membership_threshold","—")}')
        self.status.set('Scene verified. Private research assets; redistribution rights are not cleared.')
        self.canvas.focus_set();self.refresh()

    def load_scene(self,path):self._job(lambda:DesktopSession(SceneBundle.load(path)),self._adopt,'Verifying scene files…')
    def load_session(self,path):self._job(lambda:DesktopSession.open(path),self._adopt,'Verifying saved session…')

    def choose_scene(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Open verified scene bundle',filetypes=[('Scene manifest','*.json')],parent=self.root)
        if path:self.load_scene(path)

    def choose_session(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Open ReFrame session',filetypes=[('ReFrame session','*.reframe.json'),('JSON','*.json')],parent=self.root)
        if path:self.load_session(path)

    def save_session(self,path):
        if self.session and not self.busy:
            try:self.session.save(path);self.status.set(f'Session saved: {Path(path).name}')
            except Exception as error:self._error(error)

    def choose_save(self):
        if not self.session or self.busy:return
        path=filedialog.asksaveasfilename(title='Save session to a new file',defaultextension='.reframe.json',filetypes=[('ReFrame session','*.reframe.json')],parent=self.root)
        if path:self.save_session(path)

    def start_export(self,path):
        if not self.session or self.busy:return
        session=self.session
        def work():return export_session(session,path,progress=lambda i,n:self.events.put(('progress',i,n)))
        def done(report):
            self.last_export=Path(path);self.status.set(f'Export verified: {Path(path).name} · {report["frames"]} states · private research use')
        self._job(work,done,'Preparing verified MP4 export…')

    def choose_export(self):
        if not self.session or self.busy:return
        path=filedialog.asksaveasfilename(title='Export private research video to a new file',defaultextension='.mp4',filetypes=[('MP4 video','*.mp4')],parent=self.root)
        if path:self.start_export(path)

    def refresh(self):
        if self.closed:return
        if not self.session:
            self.canvas.delete('all');self.canvas.create_text(max(1,self.canvas.winfo_width())/2,max(1,self.canvas.winfo_height())/2,
                text='Open a verified scene bundle\nto select and inspect a reconstructed object.',fill=MUTED,font=('Helvetica',14),justify='center');return
        s=self.session
        try:
            with Image.open(s.scene.image_path(s.frame,s.view,s.mode)) as image:
                cw=max(1,self.canvas.winfo_width()-2);ch=max(1,self.canvas.winfo_height()-2)
                scale=min(cw/s.scene.width,ch/s.scene.height);w=max(1,int(s.scene.width*scale));h=max(1,int(s.scene.height*scale))
                photo=ImageTk.PhotoImage(image.resize((w,h),Image.Resampling.LANCZOS),master=self.root)
            self.photo=photo;self.rect=((cw-w)//2,(ch-h)//2,w,h)
            self.canvas.delete('all');self.canvas.create_image(self.rect[0],self.rect[1],anchor='nw',image=photo)
        except Exception as error:self.stop_play();self._error(error);return
        self.object_button.configure(text=('✓  ' if s.selected else '')+s.scene.data['object']['label'])
        self.mode_var.set(s.mode);self.view_var.set(next(k for k,v in VIEW_LABELS.items() if v==s.view))
        self.timeline.set(s.frame);self.frame_label.configure(text=f'State {s.frame+1:02d} / {s.scene.frames:02d}  ·  {s.scene.fps:g} fps')
        self.selection_label.configure(text=('Selected: '+s.scene.data['object']['label']) if s.selected else 'Click the balloon or select it in the object list.')
        self._controls()

    def resize(self,event=None):
        if self.refresh_timer:self.root.after_cancel(self.refresh_timer)
        self.refresh_timer=self.root.after(30,self.refresh)

    def toggle_selection(self):
        if self.session and not self.busy:self.session.select(not self.session.selected);self.refresh()

    def viewport_click(self,event):
        if not self.session or self.busy:return
        self.canvas.focus_set()
        s=self.session;point=viewport_pixel(event.x,event.y,self.rect,(s.scene.width,s.scene.height))
        if point is None:return
        try:s.select(s.scene.hit_test(s.frame,s.view,*point));self.refresh()
        except Exception as error:self._error(error)

    def change_mode(self):
        if self.session and not self.busy:
            try:self.session.set_mode(self.mode_var.get());self.refresh()
            except Exception as error:self._error(error)

    def change_view(self,event=None):
        if self.session and not self.busy:self.session.set_view(VIEW_LABELS[self.view_var.get()]);self.refresh()

    def scrub(self,value):
        if self.session and not self.busy and int(float(value))!=self.session.frame:
            self.stop_play();self.session.seek(int(float(value)));self.refresh()

    def step(self,delta):
        if self.session and not self.busy:self.stop_play();self.session.seek((self.session.frame+delta)%self.session.scene.frames);self.refresh()

    def undo(self):
        if self.session and not self.busy:self.session.undo();self.refresh()
    def reset(self):
        if self.session and not self.busy:self.session.reset();self.refresh()

    def stop_play(self):
        self.playing=False
        if self.play_timer:self.root.after_cancel(self.play_timer);self.play_timer=None
        self.play_button.configure(text='Play')

    def toggle_play(self):
        if not self.session or self.busy:return
        if self.playing:self.stop_play()
        else:self.playing=True;self.play_button.configure(text='Pause');self._advance()

    def _advance(self):
        if not self.playing or self.closed:return
        self.session.tick();self.refresh();self.play_timer=self.root.after(round(1000/self.session.scene.fps),self._advance)

    def close(self):
        if self.busy:
            messagebox.showinfo('ReFrame','Wait for verification/export to finish before closing.',parent=self.root);return
        self.closed=True;self.stop_play()
        for timer in [self.poll_timer,self.refresh_timer]:
            if timer:
                try:self.root.after_cancel(timer)
                except tk.TclError:pass
        self.executor.shutdown(wait=False,cancel_futures=True);self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',type=Path)
    args=parser.parse_args();root=tk.Tk();DesktopApp(root,args.scene);root.mainloop()


if __name__=='__main__':main()
