import cv2 as cv
from utils.face_recognition import detect_face
from utils.motion_tracking import track_gesture, is_wanted_gesture
import os
from utils.overlay_png import overlay_png


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

                frame = overlay_png(frame, icon_resized, icon_x, icon_y)

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
