import argparse
import ctypes
import pathlib
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam


CAMERA_CANDIDATES = (0, 1, 2, 3, 4)
BACKEND_CANDIDATES = (
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("AUTO", cv2.CAP_ANY),
)
WINDOW_NAME = "Preview - Unity Capture"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_FPS = 24.0
TRANSITION_SECONDS = 0.30
GESTURE_FRAME_SKIP = 2
GESTURE_HOLD_SECONDS = 0.5
GESTURE_COOLDOWN_SECONDS = 1.0
GESTURE_INPUT_SIZE = 384
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - value) ** 3


def try_read_frame(camera: cv2.VideoCapture):
    for _ in range(20):
        ok, frame = camera.read()
        if ok and frame is not None:
            return frame
        cv2.waitKey(1)

    return None


class LatestFrameCamera:
    def __init__(self, camera: cv2.VideoCapture) -> None:
        self._camera = camera
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while self._running:
            ok, frame = self._camera.read()
            if not ok or frame is None:
                continue

            with self._lock:
                self._frame = frame

    def get_latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def configure_camera(camera: cv2.VideoCapture) -> None:
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)


class GlobalHotkeys:
    def __init__(self) -> None:
        self._last_state: dict[int, bool] = {}

    def was_pressed(self, vk_code: int) -> bool:
        is_down = bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        was_down = self._last_state.get(vk_code, False)
        self._last_state[vk_code] = is_down
        return is_down and not was_down


class SegmentationEngine:
    def get_mask(self, frame_bgr: np.ndarray, background_bgr: np.ndarray | None) -> np.ndarray:
        if background_bgr is None:
            return np.ones(frame_bgr.shape[:2], dtype=np.float32)

        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.absdiff(current_gray, background_gray)

        current_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        background_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
        sat_diff = cv2.absdiff(current_hsv[:, :, 1], background_hsv[:, :, 1])
        value_diff = cv2.absdiff(current_hsv[:, :, 2], background_hsv[:, :, 2])

        combined = cv2.addWeighted(gray_diff, 0.55, sat_diff, 0.25, 0.0)
        combined = cv2.addWeighted(combined, 1.0, value_diff, 0.20, 0.0)

        _, binary_mask = cv2.threshold(combined, 24, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        _, hard_mask = cv2.threshold(binary_mask, 240, 255, cv2.THRESH_BINARY)
        edge_mask = cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
        mask = np.maximum(hard_mask, edge_mask).astype(np.float32) / 255.0
        return mask.clip(0.0, 1.0)


def ensure_model(model_path: pathlib.Path, model_url: str) -> pathlib.Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Descargando modelo: {model_path.name}")
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


class GokuGestureDetector:
    def __init__(self) -> None:
        face_model_path = ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
        hand_model_path = ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)

        base_options_face = mp.tasks.BaseOptions(model_asset_path=str(face_model_path))
        base_options_hand = mp.tasks.BaseOptions(model_asset_path=str(hand_model_path))

        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        self._hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                num_hands=1,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        )

        self._frame_counter = 0
        self._hold_started_at: float | None = None
        self._last_trigger_at = 0.0
        self._hold_triggered = False
        self._last_status = "Sin evaluar"

    @property
    def last_status(self) -> str:
        return self._last_status

    def _is_index_extended(self, hand_landmarks) -> bool:
        tip = hand_landmarks[8]
        pip = hand_landmarks[6]
        mcp = hand_landmarks[5]
        return tip.y < pip.y < mcp.y

    def _prepare_gesture_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        side = max(height, width)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)

        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        canvas[offset_y:offset_y + height, offset_x:offset_x + width] = frame_bgr

        if side != GESTURE_INPUT_SIZE:
            canvas = cv2.resize(canvas, (GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        return canvas

    def _is_pointing_to_forehead(self, frame_bgr: np.ndarray) -> bool:
        gesture_bgr = self._prepare_gesture_frame(frame_bgr)
        frame_rgb = cv2.cvtColor(gesture_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        face_result = self._face_landmarker.detect(mp_image)
        hand_result = self._hand_landmarker.detect(mp_image)

        if not face_result.face_landmarks or not hand_result.hand_landmarks:
            if not face_result.face_landmarks and not hand_result.hand_landmarks:
                self._last_status = "Sin cara ni mano"
            elif not face_result.face_landmarks:
                self._last_status = "Sin cara"
            else:
                self._last_status = "Sin mano"
            return False

        face_landmarks = face_result.face_landmarks[0]
        hand_landmarks = hand_result.hand_landmarks[0]

        if not self._is_index_extended(hand_landmarks):
            self._last_status = "Indice no extendido"
            return False

        index_tip = hand_landmarks[8]
        index_pip = hand_landmarks[6]
        forehead = face_landmarks[10]
        nose_bridge = face_landmarks[168]
        left_face = face_landmarks[234]
        right_face = face_landmarks[454]
        chin = face_landmarks[152]

        face_width = max(abs(right_face.x - left_face.x), 0.08)
        face_height = max(abs(chin.y - forehead.y), 0.12)
        finger_length = max((((index_tip.x - index_pip.x) ** 2 + (index_tip.y - index_pip.y) ** 2) ** 0.5), 0.01)
        delta_x = abs(index_tip.x - forehead.x)
        delta_y = abs(index_tip.y - forehead.y)

        within_forehead_column = delta_x < max(face_width * 0.30, finger_length * 2.8)
        within_forehead_band = delta_y < max(face_height * 0.24, finger_length * 3.0)
        above_eye_line = index_tip.y < nose_bridge.y + face_height * 0.18
        not_too_high = index_tip.y > forehead.y - face_height * 0.35

        if not within_forehead_column:
            self._last_status = "Dedo lejos en X"
            return False

        if not within_forehead_band:
            self._last_status = "Dedo lejos en Y"
            return False

        if not above_eye_line:
            self._last_status = "Dedo demasiado bajo"
            return False

        if not not_too_high:
            self._last_status = "Dedo demasiado alto"
            return False

        self._last_status = "Gesto OK"
        return True

    def should_trigger(self, frame_bgr: np.ndarray) -> bool:
        self._frame_counter += 1
        if self._frame_counter % GESTURE_FRAME_SKIP != 0:
            return False

        now = time.perf_counter()
        if now - self._last_trigger_at < GESTURE_COOLDOWN_SECONDS:
            return False

        is_match = self._is_pointing_to_forehead(frame_bgr)
        if is_match:
            if self._hold_started_at is None:
                self._hold_started_at = now

            hold_elapsed = now - self._hold_started_at
            self._last_status = f"{self._last_status} ({hold_elapsed:.2f}/{GESTURE_HOLD_SECONDS:.2f}s)"
        else:
            self._hold_started_at = None
            self._hold_triggered = False

        hold_elapsed = 0.0 if self._hold_started_at is None else now - self._hold_started_at
        triggered = hold_elapsed >= GESTURE_HOLD_SECONDS and not self._hold_triggered

        if triggered:
            self._last_trigger_at = now
            self._hold_started_at = None
            self._hold_triggered = True
            self._last_status = "Gesto disparado"

        return triggered

    def close(self) -> None:
        self._face_landmarker.close()
        self._hand_landmarker.close()


def open_camera() -> tuple[cv2.VideoCapture, int, str]:
    errors: list[str] = []

    for index in CAMERA_CANDIDATES:
        for backend_name, backend_id in BACKEND_CANDIDATES:
            camera = cv2.VideoCapture(index, backend_id)
            if not camera.isOpened():
                camera.release()
                errors.append(f"camara {index} con {backend_name}: no abre")
                continue

            configure_camera(camera)

            frame = try_read_frame(camera)
            if frame is not None:
                return camera, index, backend_name

            camera.release()
            errors.append(f"camara {index} con {backend_name}: abre pero no entrega frames")

    raise RuntimeError(
        "No se pudo abrir ninguna webcam utilizable. "
        "Cierra Teams, Discord, Zoom, OBS o la app del fabricante de la camara.\n"
        + "\n".join(errors)
    )


def read_first_frame(camera: cv2.VideoCapture) -> tuple[int, int, float]:
    frame = try_read_frame(camera)
    if frame is None:
        raise RuntimeError("No se pudo leer el primer frame de la webcam.")

    height, width = frame.shape[:2]
    fps = camera.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    return width, height, fps


def open_virtual_camera(width: int, height: int, fps: float) -> pyvirtualcam.Camera:
    errors: list[str] = []

    for backend in ("unitycapture", None):
        try:
            if backend:
                return pyvirtualcam.Camera(width=width, height=height, fps=fps, backend=backend)
            return pyvirtualcam.Camera(width=width, height=height, fps=fps)
        except Exception as error:
            label = backend or "auto"
            errors.append(f"{label}: {error}")

    raise RuntimeError(
        "No se pudo abrir la camara virtual. Comprueba que Unity Capture este instalado.\n"
        + "\n".join(errors)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Muestra una ventana de preview")
    return parser.parse_args()


def blend_frames(frame_a: np.ndarray, frame_b: np.ndarray, amount: float) -> np.ndarray:
    amount = max(0.0, min(1.0, amount))
    output = frame_a.astype(np.float32) * (1.0 - amount) + frame_b.astype(np.float32) * amount
    return np.clip(output, 0, 255).astype(np.uint8)


def get_person_center(person_mask: np.ndarray) -> tuple[int, int]:
    points = np.argwhere(person_mask > 0.35)
    if points.size == 0:
        height, width = person_mask.shape[:2]
        return width // 2, height // 2

    center_y = int(points[:, 0].mean())
    center_x = int(points[:, 1].mean())
    return center_x, center_y


def composite_person_over_background(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    person_mask: np.ndarray,
) -> np.ndarray:
    mask_3 = person_mask[..., None]
    output = frame_bgr.astype(np.float32) * mask_3 + background_bgr.astype(np.float32) * (1.0 - mask_3)
    return np.clip(output, 0, 255).astype(np.uint8)


def build_effect_mask(person_mask: np.ndarray, amount: float) -> np.ndarray:
    amount = max(0.0, min(1.0, amount))
    binary_mask = np.clip(person_mask * 255.0, 0, 255).astype(np.uint8)
    kernel_size = max(7, int(21 + amount * 16))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(binary_mask, kernel, iterations=1)
    feather = cv2.GaussianBlur(expanded, (0, 0), sigmaX=6.0 + amount * 4.0, sigmaY=6.0 + amount * 4.0)
    return feather.astype(np.float32) / 255.0


def composite_effect_inside_mask(
    base_frame_bgr: np.ndarray,
    effect_frame_bgr: np.ndarray,
    effect_mask: np.ndarray,
) -> np.ndarray:
    mask_3 = effect_mask[..., None]
    output = base_frame_bgr.astype(np.float32) * (1.0 - mask_3) + effect_frame_bgr.astype(np.float32) * mask_3
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_glitch(frame_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    band_height = max(6, height // 18)
    band_shift = max(1, int(18 * amount))
    channel_shift = max(1, int(8 * amount))

    output[:, :, 0] = np.roll(output[:, :, 0], channel_shift, axis=1)
    output[:, :, 2] = np.roll(output[:, :, 2], -channel_shift, axis=1)

    for band_index, start_y in enumerate(range(0, height, band_height)):
        end_y = min(height, start_y + band_height)
        if band_index % 2 == 0:
            output[start_y:end_y] = np.roll(output[start_y:end_y], band_shift, axis=1)
        elif band_index % 3 == 0:
            output[start_y:end_y] = np.roll(output[start_y:end_y], -band_shift, axis=1)

    scan_alpha = min(0.35, 0.22 * amount + 0.05)
    output[::2] = np.clip(output[::2].astype(np.float32) * (1.0 - scan_alpha), 0, 255).astype(np.uint8)
    return output


def apply_zoom_pulse(frame_bgr: np.ndarray, amount: float, center_x: int, center_y: int) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    scale = 1.0 + 0.08 * amount
    scaled_width = max(int(width * scale), width)
    scaled_height = max(int(height * scale), height)
    resized = cv2.resize(frame_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    center_x_scaled = int(center_x * scale)
    center_y_scaled = int(center_y * scale)

    start_x = max(0, min(center_x_scaled - width // 2, scaled_width - width))
    start_y = max(0, min(center_y_scaled - height // 2, scaled_height - height))
    return resized[start_y:start_y + height, start_x:start_x + width]


def apply_motion_smear(frame_bgr: np.ndarray, amount: float, direction: int) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    output = frame_bgr.astype(np.float32) * 0.42
    shifts = [
        max(1, int(10 * amount)),
        max(2, int(22 * amount)),
        max(3, int(34 * amount)),
    ]
    weights = (0.24, 0.18, 0.12)

    for shift, weight in zip(shifts, weights):
        output += np.roll(frame_bgr, shift * direction, axis=1).astype(np.float32) * weight
        output += np.roll(frame_bgr, -shift * direction, axis=1).astype(np.float32) * (weight * 0.55)

    return np.clip(output, 0, 255).astype(np.uint8)


def apply_vertical_phase_warp(frame_bgr: np.ndarray, amount: float, center_x: int) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    slice_width = max(3, width // 42)
    max_shift = max(2, int(18 * amount))

    for start_x in range(0, width, slice_width):
        end_x = min(width, start_x + slice_width)
        distance = abs((start_x + end_x) // 2 - center_x) / max(width, 1)
        local_strength = max(0.0, 1.0 - distance * 2.1)
        if local_strength <= 0.0:
            continue

        shift = int(np.sin((start_x / max(slice_width, 1)) * 1.25) * max_shift * local_strength)
        output[:, start_x:end_x] = np.roll(output[:, start_x:end_x], shift, axis=0)

    return output


def draw_goku_energy_lines(
    frame_bgr: np.ndarray,
    amount: float,
    center_x: int,
    center_y: int,
    person_mask: np.ndarray,
) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    line_alpha = min(0.40, 0.16 + amount * 0.22)
    color = np.array([255, 245, 210], dtype=np.float32)
    overlay = output.astype(np.float32)
    binary_mask = (person_mask > 0.25).astype(np.uint8)
    columns = np.where(binary_mask.sum(axis=0) > max(3, height // 22))[0]
    if columns.size == 0:
        return output

    step = max(6, len(columns) // 14)
    thickness = max(1, int(1 + amount * 2))

    for index in range(0, len(columns), step):
        x = int(columns[index])
        column_rows = np.where(binary_mask[:, x] > 0)[0]
        if column_rows.size == 0:
            continue

        top_y = int(column_rows.min())
        bottom_y = int(column_rows.max())
        extension = max(6, int(20 * amount))
        cv2.line(
            overlay,
            (x, max(0, top_y - extension)),
            (x, min(height - 1, bottom_y + extension)),
            color.tolist(),
            thickness,
            cv2.LINE_AA,
        )

    accent_alpha = min(0.22, amount * 0.18)
    cv2.line(
        overlay,
        (max(0, center_x), max(0, center_y - int(70 * amount))),
        (max(0, center_x), min(height - 1, center_y + int(70 * amount))),
        [255, 255, 255],
        max(1, thickness),
        cv2.LINE_AA,
    )

    combined = cv2.addWeighted(overlay.astype(np.uint8), line_alpha, output, 1.0 - line_alpha, 0.0)
    return cv2.addWeighted(combined, 1.0, output, accent_alpha, 0.0)


def apply_flash(frame_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    white = np.full_like(frame_bgr, 255)
    flash_alpha = min(0.82, amount * 0.9)
    return blend_frames(frame_bgr, white, flash_alpha)


def apply_goku_tint(frame_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    tint = np.full_like(frame_bgr, (120, 205, 255))
    tinted = blend_frames(frame_bgr, tint, min(0.20, amount * 0.18))
    highlight = np.full_like(frame_bgr, (210, 245, 255))
    return blend_frames(tinted, highlight, min(0.14, amount * 0.10))


def apply_outline_flash(frame_bgr: np.ndarray, person_mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame_bgr

    binary_mask = np.clip(person_mask * 255.0, 0, 255).astype(np.uint8)
    outer = cv2.dilate(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    inner = cv2.erode(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
    outline = cv2.subtract(outer, inner)
    outline = cv2.GaussianBlur(outline, (0, 0), sigmaX=1.8, sigmaY=1.8).astype(np.float32) / 255.0

    mask_3 = outline[..., None] * min(0.85, amount * 0.9)
    glow = np.full_like(frame_bgr, (240, 250, 255)).astype(np.float32)
    output = frame_bgr.astype(np.float32) * (1.0 - mask_3) + glow * mask_3
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_instant_transmission_transition(
    visible_frame_bgr: np.ndarray,
    hidden_frame_bgr: np.ndarray,
    person_mask: np.ndarray,
    progress: float,
    center_x: int,
    center_y: int,
    direction_out: bool,
) -> np.ndarray:
    progress = max(0.0, min(1.0, progress))
    mix_amount = ease_out_cubic(progress)
    base_frame = blend_frames(visible_frame_bgr, hidden_frame_bgr, mix_amount)

    pulse_amount = max(0.0, 1.0 - abs(progress - 0.10) / 0.10)
    flash_amount = max(0.0, 1.0 - abs(progress - 0.16) / 0.08)
    smear_amount = max(0.0, 1.0 - abs(progress - 0.20) / 0.12)
    warp_amount = max(0.0, 1.0 - abs(progress - 0.18) / 0.12)
    line_amount = max(0.0, 1.0 - abs(progress - 0.15) / 0.10)
    glitch_amount = max(0.0, 1.0 - abs(progress - 0.18) / 0.10)
    outline_amount = max(0.0, 1.0 - abs(progress - 0.14) / 0.08)

    direction = 1 if direction_out else -1
    local_effect = apply_zoom_pulse(base_frame, pulse_amount * 0.55, center_x, center_y)
    local_effect = apply_vertical_phase_warp(local_effect, warp_amount * 0.65, center_x)
    local_effect = apply_motion_smear(local_effect, smear_amount * 0.28, direction)
    local_effect = apply_glitch(local_effect, min(0.35, glitch_amount + 0.04))
    local_effect = draw_goku_energy_lines(local_effect, line_amount, center_x, center_y, person_mask)
    local_effect = apply_goku_tint(local_effect, max(flash_amount, line_amount) * 0.65)
    local_effect = apply_outline_flash(local_effect, person_mask, outline_amount * 0.85)
    local_effect = apply_flash(local_effect, flash_amount * 0.72)

    effect_strength = max(pulse_amount, flash_amount, smear_amount, warp_amount, line_amount, glitch_amount, outline_amount)
    effect_mask = build_effect_mask(person_mask, effect_strength)
    return composite_effect_inside_mask(base_frame, local_effect, effect_mask)


def draw_status_overlay(
    frame_bgr: np.ndarray,
    state_name: str,
    background_ready: bool,
    gesture_enabled: bool,
    gesture_status: str,
    is_preview: bool,
) -> np.ndarray:
    if not is_preview:
        return frame_bgr

    output = frame_bgr.copy()
    lines = [
        f"Estado: {state_name}",
        f"Fondo limpio: {'SI' if background_ready else 'NO'}",
        f"Gesto Goku: {'SI' if gesture_enabled else 'NO'}",
        f"Debug gesto: {gesture_status}",
        "F6 = capturar fondo",
        "F7 = borrar fondo",
        "F8 = desaparecer",
        "F9 = reaparecer",
        "Q = salir preview",
    ]

    y = 26
    for line in lines:
        cv2.putText(
            output,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return output


def main() -> int:
    args = parse_args()
    camera, camera_index, backend_name = open_camera()
    segmenter = SegmentationEngine()
    gesture_detector = GokuGestureDetector()
    hotkeys = GlobalHotkeys()

    try:
        width, height, fps = read_first_frame(camera)
        latest_camera = LatestFrameCamera(camera)
        with open_virtual_camera(width, height, fps) as virtual_camera:
            print(f"[OK] Webcam: indice {camera_index} con backend {backend_name}")
            print(f"[OK] Resolucion: {width}x{height} @ {fps:.2f} FPS")
            print(f"[OK] Camara virtual: {virtual_camera.device}")
            print("[INFO] Abre Discord o Teams y elige esa camara.")
            print("[INFO] F6 = capturar fondo limpio estando fuera del plano.")
            print("[INFO] F7 = borrar fondo limpio.")
            print("[INFO] F8 = teletransportarse.")
            print("[INFO] F9 = reaparecer.")
            print("[INFO] Gesto Goku activo: indice estirado tocando la frente durante 0.5s. Funciona como toggle.")
            print("[INFO] Pulsa Ctrl+C para salir.")

            frame_counter = 0
            started_at = time.perf_counter()
            background_frame = None
            effect_state = "normal"
            transition_started_at = 0.0

            while True:
                frame = latest_camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                frame_counter += 1
                frame = cv2.flip(frame, 1)

                if hotkeys.was_pressed(VK_F6):
                    background_frame = frame.copy()
                    print("[OK] Fondo limpio capturado.")

                if hotkeys.was_pressed(VK_F7):
                    background_frame = None
                    effect_state = "normal"
                    print("[OK] Fondo limpio borrado.")

                gesture_triggered = False
                if background_frame is not None and effect_state in ("normal", "disappeared"):
                    gesture_triggered = gesture_detector.should_trigger(frame)

                if hotkeys.was_pressed(VK_F8) or (gesture_triggered and effect_state == "normal"):
                    if background_frame is None:
                        print("[WARN] Captura antes un fondo limpio con F6.")
                    elif effect_state == "normal":
                        effect_state = "transition_out"
                        transition_started_at = time.perf_counter()
                        if gesture_triggered:
                            print("[INFO] Gesto detectado. Iniciando teletransportacion.")
                        else:
                            print("[INFO] Iniciando teletransportacion.")

                should_reappear = hotkeys.was_pressed(VK_F9) or (
                    gesture_triggered and effect_state == "disappeared"
                )
                if should_reappear and effect_state == "disappeared":
                    effect_state = "transition_in"
                    transition_started_at = time.perf_counter()
                    if gesture_triggered:
                        print("[INFO] Gesto detectado. Reapareciendo.")
                    else:
                        print("[INFO] Reapareciendo.")

                output_frame = frame
                if background_frame is not None:
                    person_mask = segmenter.get_mask(frame, background_frame)
                    center_x, center_y = get_person_center(person_mask)
                    person_on_clean_background = composite_person_over_background(
                        frame,
                        background_frame,
                        person_mask,
                    )

                    if effect_state == "normal":
                        output_frame = frame
                    elif effect_state == "transition_out":
                        progress = (time.perf_counter() - transition_started_at) / TRANSITION_SECONDS
                        output_frame = apply_instant_transmission_transition(
                            person_on_clean_background,
                            background_frame,
                            person_mask,
                            progress,
                            center_x,
                            center_y,
                            direction_out=True,
                        )
                        if progress >= 1.0:
                            effect_state = "disappeared"
                    elif effect_state == "disappeared":
                        output_frame = background_frame
                    elif effect_state == "transition_in":
                        progress = (time.perf_counter() - transition_started_at) / TRANSITION_SECONDS
                        output_frame = apply_instant_transmission_transition(
                            background_frame,
                            person_on_clean_background,
                            person_mask,
                            progress,
                            center_x,
                            center_y,
                            direction_out=False,
                        )
                        if progress >= 1.0:
                            effect_state = "normal"

                frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                virtual_camera.send(frame_rgb)

                if args.preview:
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    fps_text = f"FPS preview: {frame_counter / elapsed:.1f}"
                    preview_frame = draw_status_overlay(
                        output_frame,
                        effect_state,
                        background_frame is not None,
                        True,
                        gesture_detector.last_status,
                        args.preview,
                    )
                    cv2.putText(
                        preview_frame,
                        fps_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                try:
                    virtual_camera.sleep_until_next_frame()
                except RuntimeError:
                    break

        return 0
    finally:
        if "latest_camera" in locals():
            latest_camera.stop()
        gesture_detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[INFO] Cerrando camara virtual...")
        raise SystemExit(0)
    except Exception as error:
        print(f"[FALLO] {error}")
        raise SystemExit(1) from error
