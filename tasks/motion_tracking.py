import cv2 as cv
from utils.face_recognition import detect_face
from utils.motion_tracking import track_hand, track_gesture, is_wanted_gesture
import os
import numpy as np


class MotionTracking:

    def __init__(self):
        icon_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "light_bulb.png")
        )
        self.icon_rgba = cv.imread(
            icon_path, cv.IMREAD_UNCHANGED
        )  # Expect RGBA (with alpha)
        if self.icon_rgba is None:
            print(f"Warning: Light bulb PNG not found at {icon_path}")

    def _overlay_png(self, frame, overlay_rgba, x, y):
        # overlay_rgba: HxWx4, frame: HxWx3 (BGR)
        if overlay_rgba is None:
            return frame

        h, w = overlay_rgba.shape[:2]
        H, W = frame.shape[:2]

        # Compute ROI in frame
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)

        # If completely out of bounds, skip
        if x0 >= x1 or y0 >= y1:
            return frame

        # Corresponding crop in overlay
        ox0, oy0 = x0 - x, y0 - y
        ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

        overlay_crop = overlay_rgba[oy0:oy1, ox0:ox1]
        if overlay_crop.shape[2] == 4:
            overlay_bgr = overlay_crop[..., :3]
            alpha = overlay_crop[..., 3:].astype(float) / 255.0
        else:
            overlay_bgr = overlay_crop
            alpha = np.ones((*overlay_bgr.shape[:2], 1), dtype=float)

        roi = frame[y0:y1, x0:x1].astype(float)
        overlay_bgr = overlay_bgr.astype(float)

        # Alpha blend
        blended = alpha * overlay_bgr + (1 - alpha) * roi
        frame[y0:y1, x0:x1] = blended.astype(frame.dtype)
        return frame

    def process_frame(self, frame):
        # Your implementation comes here:
        frame, result = track_gesture(frame)
        pointing_up = is_wanted_gesture(result, "Pointing_Up")

        if pointing_up:
            print("You are pointing up!")

        faces = detect_face(frame)
        for x, y, w, h in faces:
            if os.environ.get("DEBUG", "0") == "1":
                center = (x + w // 2, y + h // 2)
                frame = cv.ellipse(
                    frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4
                )
            if pointing_up and self.icon_rgba is not None:
                # Scale icon relative to face width
                target_w = int(w * 0.5)
                scale = target_w / self.icon_rgba.shape[1]
                target_h = max(1, int(self.icon_rgba.shape[0] * scale))
                icon_resized = cv.resize(
                    self.icon_rgba, (target_w, target_h), interpolation=cv.INTER_AREA
                )

                # Position above the face with margin
                margin = 20
                icon_x = x + w // 2 - target_w // 2
                icon_y = y - target_h - margin

                frame = self._overlay_png(frame, icon_resized, icon_x, icon_y)

        return frame

    def display_label(self, frame):
        cv.putText(
            frame,
            "Motion tracking and interaction",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        return frame
