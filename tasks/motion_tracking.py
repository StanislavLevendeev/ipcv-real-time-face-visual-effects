import cv2 as cv
from utils.face_recognition import detect_face
from utils.motion_tracking import track_hand, track_gesture, is_wanted_gesture


class MotionTracking:
    def process_frame(self, frame):
        # Your implementation comes here:
        frame, result = track_gesture(frame)
        if is_wanted_gesture(result, "Pointing_Up"):
            print("You are pointing up!")
        faces = detect_face(frame)
        for x, y, w, h in faces:
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(
                frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4
            )
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
