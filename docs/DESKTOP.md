# ReFrame desktop sample workflow

This native Python/Tk application opens the verified Balloon2 reconstruction
outputs, selects the persistent Gaussian object, plays/scrubs the saved states,
switches source/nearby cameras, previews highlight/isolation/orange recolor and
exports the chosen sequence as an independently playable MP4. Session files save
selection, appearance, camera and timeline position.

**This is a saved-render desktop demo, not a live 4D renderer or arbitrary-video
editor.** Choosing appearance switches between real GPU renders created in the
Phase 2 experiment. Canvas selection uses that object's rendered visible
contribution coverage; it does not run SAM or perform a new 3D raycast. The UUID
and selected Gaussian rows remain bound to the scene. No images are painted to
simulate a new edit. Only the verified orange preset and two saved viewpoints
are offered. Clean translation failed visual acceptance and is visibly disabled.
The earlier blur experiments remain paused.

The user authorized Phase 3 desktop work on October 30. The original roadmap DOCX
is not present in this executor; its exact Phase 3 excerpt has been requested
from the coordinator. This document records the implemented workflow without
inventing milestone numbers or claiming all roadmap exit criteria are complete.

## Launch

With the existing project environment:

```bash
.venv/bin/python -m reframe.desktop --scene outputs/desktop-balloon2/scene.json
```

For a smaller desktop-only environment, install Python 3.11+ with Tk support,
NumPy/Pillow from `requirements-desktop.txt`, and FFmpeg/ffprobe with libx264 on
PATH. No Torch, CUDA, Modal or Meta credentials are needed to run the app.

```bash
python -m venv .venv-desktop
# Activate the environment using your platform's normal command.
python -m pip install -r requirements-desktop.txt
python -m reframe.desktop --scene /path/to/scene.json
```

Official Python desktop distributions commonly include Tk on Windows/macOS;
Linux distributions may package it separately as `python3-tk`. The app has been
run and tested on Linux with Tcl/Tk 9 in a local headless X display. Windows and
macOS launch/packaging have **not** been verified. There is no signed executable,
installer, server, tunnel or deployment. A private source-run ZIP includes
allowlisted application modules, dependency list, launch scripts and verified
sample renders; it was unpacked and run in a separate directory successfully,
including MP4 export without importing Torch or any cloud SDK. The current deliverable is a desktop
Python application launched from source.

Without `--scene`, use **Open scene** to choose a `scene.json` bundle. Opening a
scene verifies its exact file inventory, paths, hashes, image formats, dimensions
and persistent object identity before enabling interaction.

## Build the private sample bundle

The repository contains no benchmark frames or checkpoints. For this existing
task, the necessary real outputs are already saved. Make a separate portable
bundle without modifying them:

```bash
python -m reframe.desktop_scene \
  --renders outputs/modal-phase-two-001/selection \
  --coverage outputs/modal-cleanup-010/selection \
  --out outputs/desktop-balloon2
```

The builder validates the Phase 2 inventory and selection sidecar, requires the
same selection/checkpoints/timeline in the coverage report, and compares the
old/new baseline pixels before reusing rendered coverage. The bundle has 120
PNG files (four appearance modes and coverage, two cameras, twelve states) and
one JSON manifest. This is existing measured reconstruction data, not synthetic
UI content. All bundle files are portable together; a saved session references
the bundle's absolute location and must be reopened with that location intact.
If relocating the bundle, open its `scene.json` and save a new session.

## Controls

- Select the balloon in the object list or click its rendered visible area.
  Click background or use Reset selection to clear it. Letterboxed margins do
  not select an object.
- Choose Original appearance, Highlight selection, Isolate object or Recolor —
  orange. The three object modes are unavailable until an object is selected.
- Switch Original camera / Nearby camera. These are two saved calibrated views,
  not free camera orbit controls.
- Play/Pause, previous/next state or timeline scrub. With the preview focused, Space toggles playback;
  left/right arrow keys step. Focused buttons retain native keyboard activation. Twelve states play at a chosen 3 fps, not an
  established acquisition cadence. No intermediate states are synthesized.
- Undo appearance (`Ctrl+Z`) restores the previous selection/appearance while
  leaving the current time and camera unchanged. Reset restores baseline.
- Save session (`Ctrl+S`) writes a new `.reframe.json`; Open session restores it
  after validating the bundle digest and object identity. Existing files are
  never silently overwritten.
- Export MP4 (`Ctrl+E`) exports **all saved states in the current camera and
  appearance mode**, not merely the current still. Scene verification and export
  run in a worker so the window remains responsive. Controls are disabled until
  the operation completes; closing waits for that operation to finish.

Export snapshots verified input frames into a temporary local directory,
encodes H264 and fully decodes it, checks frame count/duration, then publishes a
new MP4 and `.export.json` sidecar. The sidecar binds object, scene, input hashes,
mode, view, playback rate and video digest. Existing outputs are preserved.
Filesystem hard-link support is required for exclusive publication in the same
directory; unsupported destinations fail safely. Failed encoding publishes no
final video. Disk/codec errors are shown in the app.

## Evidence and verification

Tests cover missing/corrupt/path-traversing bundles, wrong dimensions, changed
frames after load, hit-testing, letterbox mapping, invalid modes, persistent
identity through timeline changes, undo, session roundtrip and stale binding,
real FFmpeg encode/decode, failure/no-overwrite behavior, and actual native widget
interaction. The GUI test is skipped when no display is available.

```bash
python -m unittest discover -s tests -p 'test_desktop_*.py'
# In an environment with a display, the same command also runs native UI events.
```

A local Linux visual QA run loaded the real Balloon2 bundle, selected via a
canvas click, previewed highlight/isolation/recolor, switched to the nearby
camera, played/paused, scrubbed to the last state, saved/reset/reopened a session,
and exported the nearby recolor MP4. Screenshots and a hash-bound QA record are
private task artifacts under `outputs/desktop-qa`. The four-second export has
12 H264 frames at 480×270. This validates the desktop workflow; it adds no new
reconstruction, selection-quality or geometry result beyond [Phase 2](PHASE2.md).

## Rights and scope

Balloon2 footage/checkpoint/output redistribution rights remain unverified.
The app labels it a private research sample. Do not commit or publicly release
the bundle, screenshots, demonstration archive or exported footage. The source
code/tests/docs may be pushed under the user's existing authorization; no
hackathon entry, public video, installer or release is being published.

Python/Tkinter, Tcl/Tk, NumPy and Pillow retain their own licenses. FFmpeg is a
separate dependency whose license depends on its build; this workflow requires
libx264. No third-party binaries are bundled. MoSca's core MIT license does not
remove its rasterizer's research/noncommercial restrictions or the benchmark
asset restrictions. See [FEASIBILITY.md](FEASIBILITY.md) and
[PUBLISHED_SAMPLE.md](PUBLISHED_SAMPLE.md). Private demonstration does not grant
permission for public asset distribution.


To package the same private demonstration after validating its bundle:

```bash
python scripts/package_desktop_demo.py --scene outputs/desktop-balloon2/scene.json \
  --out outputs/reframe-desktop-demo.zip
```

The archive includes a hash inventory and private-use notice. It is never pushed
or publicly released. Dependencies/binaries are installed separately. Session
files and private export paths are not included in the archive.
