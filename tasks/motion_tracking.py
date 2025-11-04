import cv2 as cv
from utils.motion_tracking import track_gesture, is_wanted_gesture
import os
from utils.overlay_png import overlay_png
from tasks.face_effects import FaceEffects


class MotionTracking:
    MARGIN = 20

    def __init__(self):
        icon_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "light_bulb.png")
        )
        self.icon_rgba = cv.imread(
            icon_path, cv.IMREAD_UNCHANGED
        )  # Expect RGBA (with alpha)
        if self.icon_rgba is None:
            print(f"Warning: Light bulb PNG not found at {icon_path}")
        self.face_effect = FaceEffects()

    def process_frame(self, frame):
        # Your implementation comes here:
        frame, result = track_gesture(frame)
        pointing_up = is_wanted_gesture(result, "Pointing_Up")

        if not pointing_up:
            return frame

        frame_copy = self.face_effect.process_frame(frame)
        faces = self.face_effect.last_detected_faces

        # faces is a MediaPipe FaceMesh result object (from face_mesh.process(rgb))
        # convert detected face landmarks to bounding boxes (x,y,w,h) in pixels
        if faces is None or not getattr(faces, "multi_face_landmarks", None):
            return frame_copy

        img_h, img_w = frame_copy.shape[:2]
        for face_landmarks in faces.multi_face_landmarks:
            # collect normalized coordinates
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]

            x_min = int(max(0, min(xs) * img_w))
            x_max = int(min(img_w, max(xs) * img_w))
            y_min = int(max(0, min(ys) * img_h))
            y_max = int(min(img_h, max(ys) * img_h))

            w_box = x_max - x_min
            h_box = y_max - y_min
            if w_box <= 0 or h_box <= 0:
                continue

            if os.environ.get("DEBUG", "0") == "1":
                center = (x_min + w_box // 2, y_min + h_box // 2)
                frame_copy = cv.ellipse(
                    frame_copy,
                    center,
                    (w_box // 2, h_box // 2),
                    0,
                    0,
                    360,
                    (255, 0, 255),
                    4,
                )

            if pointing_up and self.icon_rgba is not None:
                frame_copy = self.display_light_bulb(
                    frame_copy, x_min, y_min, w_box, h_box
                )

        return frame_copy

    def display_light_bulb(self, frame, x, y, w, h):
        print("You are pointing up and the face is detected")
        # Scale icon relative to face width
        target_w = int(w * 0.5)
        scale = target_w / self.icon_rgba.shape[1]
        target_h = max(1, int(self.icon_rgba.shape[0] * scale))
        icon_resized = cv.resize(
            self.icon_rgba, (target_w, target_h), interpolation=cv.INTER_AREA
        )

        icon_x = x + w // 2 - target_w // 2
        icon_y = y - target_h - self.MARGIN

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
