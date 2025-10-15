import cv2 as cv


class FaceEffects:
    def process_frame(self, frame):
        # Your implementation comes here:

        return frame

    def display_label(self, frame):
        cv.putText(
            frame,
            "Face effects",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        return frame
