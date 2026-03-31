"""actions/gesture_control.py
Hand Gesture Control for Mark-XXX using MediaPipe Tasks API (hand_landmarker).

Full gesture map
────────────────────────────────────────────────────────────────────────────────
 Gesture               Trigger                          Action
 ─────────────────     ──────────────────────────────   ────────────────────────
 POINT                 Solo índice extendido            Mueve el cursor (EMA)
 PINCH                 Pulgar + índice juntos           Click izquierdo
 RIGHT_CLICK           Pulgar + medio juntos            Click derecho
 SCROLL                Índice + medio arriba            Scroll vertical
 PALM                  5 dedos abiertos                 Captura de pantalla
 THUMB_UP              Solo pulgar arriba               Toggle mute/unmute Jarvis
 THREE_FINGERS         Índice + medio + anular arriba   Siguiente canción (media)
 PINKY_UP              Solo meñique arriba              Play / Pause media
 FIST (2 s)            Puño cerrado 2 s                 Detiene el control +
                                                        notifica a Jarvis
────────────────────────────────────────────────────────────────────────────────

Runs on a daemon thread; OpenCV preview window lives on that same thread so
the main Tkinter loop stays unblocked.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pyautogui

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
MODEL_PATH  = BASE_DIR / "config" / "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Tunable constants ──────────────────────────────────────────────────────────
PINCH_THRESHOLD      = 0.07   # normalised thumb-index distance → left click
RIGHT_CLICK_THRESH   = 0.07   # normalised thumb-middle distance → right click
FIST_HOLD_TIME       = 2.0    # seconds to hold fist before stopping
CLICK_COOLDOWN       = 0.6    # minimum seconds between any click events
SCREENSHOT_COOLDOWN  = 2.0    # minimum seconds between screenshots
SCROLL_COOLDOWN      = 0.05   # minimum seconds between scroll events
MEDIA_COOLDOWN       = 1.5    # minimum seconds between media key events
MUTE_COOLDOWN        = 1.0    # minimum seconds between mute toggles
SCROLL_SPEED         = 10     # scroll multiplier
EMA_ALPHA            = 0.25   # cursor smoothing (0 = very smooth, 1 = raw)

# Active zone: fraction of camera frame mapped to full screen.
ACTIVE_ZONE = (0.10, 0.10, 0.90, 0.90)   # (x_min, y_min, x_max, y_max)

# ── MediaPipe hand landmark indices ───────────────────────────────────────────
WRIST      = 0
THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_TIP  = 20

# Skeleton connections for drawing
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Singleton state ────────────────────────────────────────────────────────────
_controller_thread: threading.Thread | None = None
_stop_event:        threading.Event  | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_camera_index() -> int:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return int(json.load(f).get("camera_index", 0))
    except Exception:
        return 0


def _ensure_model() -> Path:
    """Download hand_landmarker.task once if it is missing."""
    if not MODEL_PATH.exists():
        print("[GESTURE] Downloading hand_landmarker.task model…")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("[GESTURE] Model downloaded successfully.")
    return MODEL_PATH


def _lm_xy(landmarks, idx: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x, lm.y


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _finger_up(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """True when the fingertip is meaningfully above (lower y) its PIP joint."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02


def _thumb_up(landmarks) -> bool:
    """Thumb is extended when its tip is far from the MCP knuckle."""
    tip = _lm_xy(landmarks, THUMB_TIP)
    mcp = _lm_xy(landmarks, THUMB_MCP)
    return _dist(tip, mcp) > 0.07


def _classify_gesture(landmarks) -> tuple[str, float, float]:
    """
    Returns (gesture_name, pinch_dist, right_click_dist).

    gesture_name ∈ {
        "POINT", "PINCH", "RIGHT_CLICK", "SCROLL",
        "PALM", "THUMB_UP", "THREE_FINGERS", "PINKY_UP",
        "FIST", "IDLE"
    }
    """
    idx_up   = _finger_up(landmarks, INDEX_TIP,  INDEX_PIP)
    mid_up   = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up  = _finger_up(landmarks, RING_TIP,   RING_PIP)
    pinky_up = _finger_up(landmarks, PINKY_TIP,  PINKY_PIP)
    t_up     = _thumb_up(landmarks)

    t_tip  = _lm_xy(landmarks, THUMB_TIP)
    i_tip  = _lm_xy(landmarks, INDEX_TIP)
    m_tip  = _lm_xy(landmarks, MIDDLE_TIP)

    pinch_dist = _dist(t_tip, i_tip)        # thumb ↔ index
    rc_dist    = _dist(t_tip, m_tip)        # thumb ↔ middle

    n_up = sum([t_up, idx_up, mid_up, ring_up, pinky_up])

    # ── Priority order ─────────────────────────────────────────────────────────
    # 1. Pinch gestures take precedence (close finger-pairs)
    if pinch_dist < PINCH_THRESHOLD:
        return "PINCH", pinch_dist, rc_dist

    if rc_dist < RIGHT_CLICK_THRESH and not idx_up:
        # thumb+middle close, index NOT extended → right click
        return "RIGHT_CLICK", pinch_dist, rc_dist

    # 2. Static count-based gestures
    if n_up == 0:
        return "FIST", pinch_dist, rc_dist

    if n_up >= 4 and t_up:
        return "PALM", pinch_dist, rc_dist

    # 3. Named single/multi-finger gestures
    # THUMB_UP: only thumb raised, all other fingers down
    if t_up and not idx_up and not mid_up and not ring_up and not pinky_up:
        return "THUMB_UP", pinch_dist, rc_dist

    # THREE_FINGERS: index + middle + ring up, pinky + thumb down
    if idx_up and mid_up and ring_up and not pinky_up and not t_up:
        return "THREE_FINGERS", pinch_dist, rc_dist

    # SCROLL: index + middle up, ring + pinky down
    if idx_up and mid_up and not ring_up and not pinky_up:
        return "SCROLL", pinch_dist, rc_dist

    # PINKY_UP: only pinky raised
    if pinky_up and not idx_up and not mid_up and not ring_up and not t_up:
        return "PINKY_UP", pinch_dist, rc_dist

    # POINT: only index raised
    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "POINT", pinch_dist, rc_dist

    return "IDLE", pinch_dist, rc_dist


def _map_to_screen(norm_x: float, norm_y: float) -> tuple[int, int]:
    """Map normalised camera coord (inside active zone) → screen pixels."""
    screen_w, screen_h = pyautogui.size()
    az = ACTIVE_ZONE
    x = (norm_x - az[0]) / (az[2] - az[0])
    y = (norm_y - az[1]) / (az[3] - az[1])
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return int(x * screen_w), int(y * screen_h)


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def _draw_skeleton(frame, landmarks, h: int, w: int, color=(0, 255, 80)):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, color, 1)


def _draw_active_zone(frame, h: int, w: int):
    az = ACTIVE_ZONE
    x1, y1 = int(az[0] * w), int(az[1] * h)
    x2, y2 = int(az[2] * w), int(az[3] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
    cv2.putText(frame, "active zone", (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)


def _draw_pinch_bar(frame, dist: float, threshold: float,
                    h: int, w: int, label: str, x_off: int, color_near):
    """Generic bar for any pinch-style distance."""
    ratio  = max(0.0, min(1.0, dist / threshold))
    bar_px = int(100 * (1.0 - ratio))    # inverted: fuller = closer to threshold
    color  = color_near if ratio < 0.6 else (80, 80, 80)
    cv2.rectangle(frame, (x_off, h - 30), (x_off + 100, h - 15), (40, 40, 40), -1)
    cv2.rectangle(frame, (x_off, h - 30), (x_off + bar_px, h - 15), color, -1)
    cv2.putText(frame, label, (x_off, h - 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def _draw_gesture_label(frame, label: str, color, h: int, w: int,
                        muted: bool):
    # Mute indicator
    mute_txt   = "  [MIC MUTED]" if muted else ""
    mute_color = (0, 50, 220) if muted else (0, 0, 0)
    full_label = label + mute_txt
    # Drop-shadow
    cv2.putText(frame, full_label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 3)
    cv2.putText(frame, label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
    if muted:
        cv2.putText(frame, mute_txt, (10 + cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)[0][0], 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (50, 80, 255), 2)


def _draw_fist_progress(frame, fist_start: float, h: int, w: int):
    prog   = min(1.0, (time.time() - fist_start) / FIST_HOLD_TIME)
    bar_px = int(w * prog)
    cv2.rectangle(frame, (0, h - 8), (bar_px, h), (30, 30, 220), -1)
    cv2.putText(frame, f"Hold fist to STOP  ({prog*100:.0f}%)", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 255), 1)


# ── Controller loop ─────────────────────────────────────────────────────────────

def _run_controller(stop_event: threading.Event, player=None):
    """Main gesture-control loop.  Runs on its own daemon thread."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        msg = "[GESTURE] mediapipe not installed.  Run: pip install mediapipe"
        print(msg)
        if player:
            player.write_log(msg)
        return

    model_path = _ensure_model()
    cam_idx    = _get_camera_index()

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        msg = f"[GESTURE] Cannot open camera index {cam_idx}. Check config/api_keys.json."
        print(msg)
        if player:
            player.write_log(msg)
        detector.close()
        return

    # ── Per-loop state ─────────────────────────────────────────────────────────
    cursor_x, cursor_y = pyautogui.position()
    last_click_t   = 0.0
    last_scroll_t  = 0.0
    last_shot_t    = 0.0
    last_media_t   = 0.0
    last_mute_t    = 0.0
    fist_start_t   = None
    prev_scroll_y  = None
    frame_ts_ms    = 0
    jarvis_muted   = False     # local shadow of Jarvis mute state

    _LABEL_MAP = {
        "POINT":        "POINT  —  moving cursor",
        "PINCH":        "PINCH  —  left click!",
        "RIGHT_CLICK":  "RIGHT CLICK  —  click derecho!",
        "SCROLL":       "SCROLL  —  scrolling",
        "PALM":         "PALM  —  screenshot",
        "THUMB_UP":     "THUMB UP  —  toggle mute",
        "THREE_FINGERS": "THREE  —  next track ⏭",
        "PINKY_UP":     "PINKY  —  play / pause ⏯",
        "FIST":         "FIST  —  hold 2 s to stop…",
        "IDLE":         "IDLE",
    }

    print("[GESTURE] Controller loop started.")
    if player:
        player.write_log("[GESTURE] Gesture control active — show hand to camera.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame       = cv2.flip(frame, 1)
        h, w        = frame.shape[:2]
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img      = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts_ms += 33       # ~30 fps

        result       = detector.detect_for_video(mp_img, frame_ts_ms)
        now          = time.time()
        gesture      = "IDLE"
        pinch_dist   = 1.0
        rc_dist      = 1.0
        skel_color   = (120, 120, 120)

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            gesture, pinch_dist, rc_dist = _classify_gesture(lms)
            skel_color = (0, 255, 80) if gesture != "IDLE" else (120, 120, 120)

            # ── POINT: move cursor ─────────────────────────────────────────
            if gesture == "POINT":
                ix, iy = _lm_xy(lms, INDEX_TIP)
                sx, sy = _map_to_screen(ix, iy)
                cursor_x = int(EMA_ALPHA * sx + (1.0 - EMA_ALPHA) * cursor_x)
                cursor_y = int(EMA_ALPHA * sy + (1.0 - EMA_ALPHA) * cursor_y)
                pyautogui.moveTo(cursor_x, cursor_y, duration=0)

            # ── PINCH: left click ──────────────────────────────────────────
            elif gesture == "PINCH":
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.click(button="left")
                    last_click_t = now

            # ── RIGHT_CLICK: right click ───────────────────────────────────
            elif gesture == "RIGHT_CLICK":
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.click(button="right")
                    last_click_t = now

            # ── SCROLL: two-finger scroll ──────────────────────────────────
            elif gesture == "SCROLL":
                iy = lms[INDEX_TIP].y
                if prev_scroll_y is not None and now - last_scroll_t > SCROLL_COOLDOWN:
                    dy = iy - prev_scroll_y       # positive → moving down
                    if abs(dy) > 0.004:
                        amount = int(dy * SCROLL_SPEED * -12)
                        pyautogui.scroll(amount)
                        last_scroll_t = now
                prev_scroll_y = iy

            # ── PALM: screenshot ───────────────────────────────────────────
            elif gesture == "PALM":
                if now - last_shot_t > SCREENSHOT_COOLDOWN:
                    import mss, datetime
                    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = Path.home() / "Desktop" / f"gesture_shot_{ts}.png"
                    with mss.mss() as sct:
                        sct.shot(output=str(out))
                    last_shot_t = now
                    msg = f"[GESTURE] Screenshot → Desktop/{out.name}"
                    print(msg)
                    if player:
                        player.write_log(msg)

            # ── THUMB_UP: toggle Jarvis mute ───────────────────────────────
            elif gesture == "THUMB_UP":
                if now - last_mute_t > MUTE_COOLDOWN:
                    jarvis_muted = not jarvis_muted
                    # Invoke the mute callback wired up by JarvisLive
                    if player and hasattr(player, "mute_callback") and callable(player.mute_callback):
                        player.mute_callback(jarvis_muted)
                    state_str = "MUTED 🔇" if jarvis_muted else "UNMUTED 🎙"
                    msg = f"[GESTURE] Jarvis microphone {state_str}"
                    print(msg)
                    if player:
                        player.write_log(msg)
                    last_mute_t = now

            # ── THREE_FINGERS: next media track ───────────────────────────
            elif gesture == "THREE_FINGERS":
                if now - last_media_t > MEDIA_COOLDOWN:
                    pyautogui.press("nexttrack")
                    last_media_t = now
                    if player:
                        player.write_log("[GESTURE] ⏭ Next track")

            # ── PINKY_UP: play / pause ─────────────────────────────────────
            elif gesture == "PINKY_UP":
                if now - last_media_t > MEDIA_COOLDOWN:
                    pyautogui.press("playpause")
                    last_media_t = now
                    if player:
                        player.write_log("[GESTURE] ⏯ Play / Pause")

            # ── FIST: hold 2 s to stop ─────────────────────────────────────
            elif gesture == "FIST":
                if fist_start_t is None:
                    fist_start_t = now
                elif now - fist_start_t >= FIST_HOLD_TIME:
                    stop_event.set()
                    # Restore mute to off so Jarvis can speak again
                    if jarvis_muted and player and hasattr(player, "mute_callback") \
                            and callable(player.mute_callback):
                        player.mute_callback(False)
                    break

            # Reset stateful trackers when gesture changes
            if gesture != "FIST":
                fist_start_t = None
            if gesture != "SCROLL":
                prev_scroll_y = None

            _draw_skeleton(frame, lms, h, w, skel_color)

        else:
            fist_start_t  = None
            prev_scroll_y = None

        # ── Overlay ────────────────────────────────────────────────────────
        _draw_active_zone(frame, h, w)

        # Two pinch bars: left click (thumb-index) and right click (thumb-middle)
        _draw_pinch_bar(frame, pinch_dist, PINCH_THRESHOLD,
                        h, w, "L-CLICK", 10,  (0, 220, 100))
        _draw_pinch_bar(frame, rc_dist, RIGHT_CLICK_THRESH,
                        h, w, "R-CLICK", 125, (0, 160, 255))

        lbl_color = (0, 255, 80) if gesture not in ("IDLE",) else (160, 160, 160)
        _draw_gesture_label(frame,
                            _LABEL_MAP.get(gesture, gesture),
                            lbl_color, h, w, jarvis_muted)

        if gesture == "FIST" and fist_start_t:
            _draw_fist_progress(frame, fist_start_t, h, w)

        cv2.putText(frame, "Mark-XXX  |  Q = quit", (w // 2 - 80, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        cv2.imshow("Mark-XXX — Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyWindow("Mark-XXX — Gesture Control")
    detector.close()
    print("[GESTURE] Controller stopped.")
    if player:
        player.write_log(
            "[GESTURE] Control gestual desactivado. "
            "Di 'Jarvis' para volver a escucharme."
        )


# ── Public API ──────────────────────────────────────────────────────────────────

class GestureController:
    """Singleton wrapper around the gesture-control daemon thread."""

    @staticmethod
    def gesture_control(parameters: dict, player=None) -> str:
        global _controller_thread, _stop_event

        action = parameters.get("action", "start").strip().lower()

        # ── STOP ──────────────────────────────────────────────────────────────
        if action == "stop":
            if _stop_event and not _stop_event.is_set():
                _stop_event.set()
                return "Gesture control detenido."
            return "El control gestual no estaba activo."

        # ── START ─────────────────────────────────────────────────────────────
        if _controller_thread and _controller_thread.is_alive():
            return "El control gestual ya está activo."

        _stop_event        = threading.Event()
        _controller_thread = threading.Thread(
            target=_run_controller,
            args=(_stop_event, player),
            daemon=True,
            name="GestureControllerThread",
        )
        _controller_thread.start()

        return (
            "Control gestual iniciado. "
            "Gestos disponibles: "
            "índice = mover cursor, "
            "pulgar+índice = click izquierdo, "
            "pulgar+medio = click derecho, "
            "índice+medio = scroll, "
            "palma abierta = captura de pantalla, "
            "pulgar arriba = silenciar o activar micrófono, "
            "tres dedos = siguiente canción, "
            "meñique = play/pause, "
            "puño cerrado 2 segundos = detener control gestual."
        )
