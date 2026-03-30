# Finger Tracking Prototype

A minimal prototype that uses your camera to detect hands and overlay a finger skeleton using MediaPipe and OpenCV.

Requirements
- Python 3.8+
- A webcam accessible by the OS

Install

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run

```powershell
# Mirror view (selfie):
python main.py --mirror

# Or normal camera view:
python main.py

# Press 'q' to quit, 'm' to toggle mirroring while running.
```

Notes
- If camera doesn't open, try changing the `--camera` index (0, 1, ...).
- On Windows, ensure the app has camera permissions in Settings -> Privacy -> Camera.
- For packaging into an .exe consider using PyInstaller:

```powershell
pip install pyinstaller
pyinstaller --onefile main.py
```

Next steps
- Add GUI controls, save frames, or stream landmarks to Unity or another consumer.
