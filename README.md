# Blink to Scroll

A hands-free, real-time scrolling system that uses your webcam and eye blinks to scroll through content — no mouse, no touch, no special hardware required.

Built with Python and MediaPipe's Face Landmarker for accurate Eye Aspect Ratio (EAR) based blink detection.

---

## How It Works

MediaPipe detects 468 facial landmarks in real time. Six specific points around each eye are used to calculate the **Eye Aspect Ratio (EAR)**. When both eyes close (EAR drops below a threshold), a blink is confirmed and a scroll event is triggered via PyAutoGUI.

```
Webcam → RGB Frame → Face Landmarks → EAR Calculation → Blink Confirmed → Scroll Down
```

---

## Features

- Real-time blink detection using MediaPipe Face Landmarker
- Eye Aspect Ratio (EAR) based detection — more accurate than Haar Cascade
- On-screen HUD showing EAR value, FPS, blink count, scroll status
- Visual EAR bar meter and blink progress dots
- Eye outline drawn on detected eyes (turns red on blink)
- Debounce logic to prevent accidental multiple scrolls
- Live sensitivity adjustment with keyboard
- Toggle scrolling on/off without restarting

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9 to 3.12 | Core language |
| MediaPipe v0.10.33 | Face landmark detection (468 points) |
| OpenCV | Webcam capture and frame display |
| PyAutoGUI | OS-level scroll simulation |
| NumPy | EAR vector math |

---

## Installation

**Step 1 — Install dependencies**
```bash
pip install mediapipe opencv-python pyautogui numpy
```

**Step 2 — Download the MediaPipe model file**

Open this URL in your browser and save the file:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**Step 3 — Place the model file in the project folder**
```
BlinkToScroll/
├── blink_to_scroll.py
├── face_landmarker.task        ← place here

```

**Step 4 — Run**
```bash
python blink_to_scroll.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the program |
| `S` | Toggle scroll on / off |
| `+` | Increase EAR threshold (less sensitive) |
| `-` | Decrease EAR threshold (more sensitive) |

---

## HUD Display

| Element | Description |
|---------|-------------|
| EAR | Current Eye Aspect Ratio value |
| THR | Current blink detection threshold |
| FPS | Frames per second |
| BLINKS | Total blinks detected in session |
| SCROLL | Whether scrolling is ON or OFF |
| OPEN / BLINK | Current eye state |
| EAR bar | Visual meter showing how open the eyes are |
| Progress dots | How many frames eyes have been closed vs required minimum |

---

## Configuration

All parameters are in the `CONFIG` dictionary at the top of the script:

```python
CONFIG = {
    "ear_threshold":     0.20,   # lower = needs more eye closure to trigger
    "min_frames_closed": 3,      # frames eyes must stay closed to confirm blink
    "debounce_time":     1.0,    # seconds between scroll triggers
    "scroll_amount":    -350,    # negative = down, positive = up
    "sensitivity_step":  0.01,   # how much +/- keys adjust the threshold
    "webcam_index":      0,      # change to 1 if using external webcam
    "frame_width":       640,
    "frame_height":      480,
}
```

---

## Performance

| Metric | Result |
|--------|--------|
| Detection method | MediaPipe EAR (468 landmarks) |
| Frame rate | ~20–30 FPS |
| Blink detection delay | ~0.3–0.5 seconds |
| CPU usage | ~12–20% |
| Detection accuracy | ~92–96% (good lighting) |

---

## Known Limitations

- Works best in well-lit, front-facing conditions
- May miss blinks in very low light or at sharp face angles
- Scroll may not work in all applications — depends on OS scroll event support
- Requires `face_landmarker.task` model file to be present locally

---

## Planned Enhancements

- [ ] Double blink to scroll up
- [ ] Left / right wink to switch tabs
- [ ] Adaptive EAR threshold based on user calibration
- [ ] GUI settings panel
- [ ] Session log export

---

## Use Cases

- Assistive technology for users with motor impairments (ALS, paralysis, arthritis)
- Touchless navigation in hospitals, labs, or cleanrooms
- Hands-free reading of documents, PDFs, or e-books
- Any scenario where physical interaction with a device is not possible

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgements

Developed under the guidance of **Mr. Akash Dixit**, Assistant Professor, Department of Science and Computing, CBSA Mohali.

Submitted as a final year project for BSc Artificial Intelligence and Machine Learning (2022–2025).
