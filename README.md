# Real-time Face Visual Effects
This project implements real-time face visual effects using computer-vision techniques (OpenCV, MediaPipe, etc.).

The project is done by the group 61:
```
Felix Reichwein     - 3755622
Joseph Tartivel     - 3723488
Stanislav Levendeev - 3544648
```

## Project structure

Top-level layout (important files and folders):

```
demo.py
README.md
requirements.txt
.env
assets/
data/
music/
tasks/
utils/
display/
```

Short descriptions

- `demo.py` — Small runner script that launches the real-time demo (camera input, processing pipeline, and on-screen display).
- `requirements.txt` — Python dependencies required to run the project.
- `.env` — Environment variables for configuration.
- `assets/` — Static assets used by effects (images, icons, stickers, etc.).
- `data/` — Models and data files used by the pipeline, for example:
	- `gesture_recognizer.task` — trained gesture recognizer/task file
- `music/` — Optional audio files used in demos or examples.
- `tasks/` — Core effect and task modules that implement processing stages:
	- `combined_task.py` — Combines multiple effects/tasks into a single pipeline
	- `face_effects.py` — Implements per-face visual effects and overlays
	- `face_warping.py` — Face warping (mesh-based) transformations
	- `motion_tracking.py` — Motion / gesture tracking utilities exposed as tasks
	- `task_manager.py` — Orchestrates tasks and their lifecycle
- `utils/` — Helper utilities used by tasks and display code:
	- `motion_tracking.py` — lower-level motion tracking utilities
	- `overlay_png.py` — helper for loading and compositing PNG overlays with alpha
    - `display/` — Presentation layer for showing output / GUI helpers:
        - `gesture_display.py` — display helpers for gesture feedback and overlays

## How to run (quick)

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Start the demo (requires a webcam):

```powershell
python demo.py
```
3. Navigate through different effects using keyboard inputs `1-9`

4. Press `d` to toggle debug mode on/off

5. Press `q` to quit the demo


Notes

- Some modules depend on camera access and pre-trained models in `data/` — ensure those files are present.
- If you add or move modules, update this section so new collaborators can find things quickly.

