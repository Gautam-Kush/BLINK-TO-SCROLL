import cv2
import time
import os
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CONFIG = {
    "model_path":        "face_landmarker.task",
    "ear_threshold":     0.20,
    "min_frames_closed": 3,
    "debounce_time":     1.0,
    "scroll_amount":    -350,
    "sensitivity_step":  0.01,
    "webcam_index":      0,
    "frame_width":       640,
    "frame_height":      480,
}

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

GREEN  = (0,   220,   0)
RED    = (0,     0, 220)
YELLOW = (0,   210, 210)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
BLACK  = (0,     0,   0)


def get_eye_coords(landmarks, indices, w, h):
    return [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]


def compute_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def draw_eye_outline(frame, eye_coords, color):
    pts = np.array([pt.astype(int) for pt in eye_coords], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
    for pt in eye_coords:
        cv2.circle(frame, tuple(pt.astype(int)), 2, color, -1)


def draw_hud(frame, ear, threshold, scroll_enabled, blink_count, closed_frames, min_frames, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    bx, by, bw, bh = 15, 90, 18, 160
    filled    = int(bh * min(ear / 0.40, 1.0))
    bar_color = RED if ear < threshold else GREEN
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), GRAY, 1)
    cv2.rectangle(frame, (bx, by + bh - filled), (bx + bw, by + bh), bar_color, -1)
    cv2.putText(frame, "EAR", (bx - 2, by + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)

    for i in range(min_frames):
        cv2.circle(frame, (50 + i * 14, 155), 5, RED if i < closed_frames else GRAY, -1)

    status_color  = RED   if ear < threshold else GREEN
    enabled_color = GREEN if scroll_enabled  else RED

    cv2.putText(frame, f"EAR:    {ear:.3f}",                              (50,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE,        1)
    cv2.putText(frame, f"THR:    {threshold:.3f}",                        (50,  42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW,       1)
    cv2.putText(frame, f"FPS:    {fps:.1f}",                              (50,  62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY,         1)
    cv2.putText(frame, f"BLINKS: {blink_count}",                          (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE,        1)
    cv2.putText(frame, f"SCROLL: {'ON' if scroll_enabled else 'OFF'}",    (220, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, enabled_color,1)
    cv2.putText(frame, "BLINK" if ear < threshold else "OPEN",            (220, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 22), (w, h), BLACK, -1)
    cv2.addWeighted(overlay2, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, "Q: Quit   S: Toggle Scroll   +/-: Sensitivity",
                (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)


class BlinkToScroll:

    def __init__(self):
        self._init_webcam()
        self._init_landmarker()

        self.ear_threshold   = CONFIG["ear_threshold"]
        self.min_frames      = CONFIG["min_frames_closed"]
        self.debounce_time   = CONFIG["debounce_time"]
        self.scroll_amount   = CONFIG["scroll_amount"]
        self.scroll_enabled  = True
        self.closed_frames   = 0
        self.last_blink_time = 0.0
        self.blink_count     = 0
        self.fps             = 0.0
        self._prev_time      = time.time()

        print("[INFO] Blink to Scroll started")
        print(f"[INFO] EAR threshold : {self.ear_threshold}")
        print(f"[INFO] Debounce time : {self.debounce_time}s")
        print("[INFO] Controls: Q=Quit  S=Toggle  +/-=Sensitivity\n")

    def _init_webcam(self):
        self.cap = cv2.VideoCapture(CONFIG["webcam_index"])
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        print("[INFO] Webcam initialised.")

    def _init_landmarker(self):
        model = CONFIG["model_path"]
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file not found: '{model}'")
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        print("[INFO] FaceLandmarker loaded successfully.")

    def _update_fps(self):
        now             = time.time()
        self.fps        = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

    def _handle_keys(self, key):
        if key == ord('q'):
            return False
        if key == ord('s'):
            self.scroll_enabled = not self.scroll_enabled
            print(f"[INFO] Scroll {'enabled' if self.scroll_enabled else 'disabled'}.")
        if key in (ord('+'), ord('=')):
            self.ear_threshold = min(self.ear_threshold + CONFIG["sensitivity_step"], 0.40)
            print(f"[INFO] EAR threshold → {self.ear_threshold:.3f}")
        if key == ord('-'):
            self.ear_threshold = max(self.ear_threshold - CONFIG["sensitivity_step"], 0.05)
            print(f"[INFO] EAR threshold → {self.ear_threshold:.3f}")
        return True

    def _process_blink(self, ear):
        if ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.min_frames:
                self._trigger_scroll()
            self.closed_frames = 0

    def _trigger_scroll(self):
        if not self.scroll_enabled:
            return
        now = time.time()
        if now - self.last_blink_time > self.debounce_time:
            pyautogui.scroll(self.scroll_amount)
            self.blink_count    += 1
            self.last_blink_time = now
            print(f"[ACTION] Blink #{self.blink_count} detected → Scrolling down")
        else:
            remaining = self.debounce_time - (now - self.last_blink_time)
            print(f"[INFO] Debounce active — {remaining:.1f}s remaining")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            self._update_fps()
            h, w         = frame.shape[:2]
            timestamp_ms = int(time.time() * 1000)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            ear = 0.0

            if results.face_landmarks:
                lm        = results.face_landmarks[0]
                left_eye  = get_eye_coords(lm, LEFT_EYE,  w, h)
                right_eye = get_eye_coords(lm, RIGHT_EYE, w, h)
                ear       = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2.0
                eye_color = RED if ear < self.ear_threshold else GREEN
                draw_eye_outline(frame, left_eye,  eye_color)
                draw_eye_outline(frame, right_eye, eye_color)
                self._process_blink(ear)
            else:
                self.closed_frames = 0
                cv2.putText(frame, "No face detected", (w // 2 - 110, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)

            draw_hud(frame, ear, self.ear_threshold, self.scroll_enabled,
                     self.blink_count, self.closed_frames, self.min_frames, self.fps)

            cv2.imshow("Blink to Scroll  |  Q to quit", frame)

            if not self._handle_keys(cv2.waitKey(1) & 0xFF):
                break

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Session ended — Total blinks detected: {self.blink_count}")


if __name__ == "__main__":
    try:
        app = BlinkToScroll()
        app.run()
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
