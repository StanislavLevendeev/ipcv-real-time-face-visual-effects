"""Standalone smoke test for the FaceWarping effect.

Runs FaceWarping on its own (no TaskManager, no other effects) so you can
iterate on jaw/beard/eyebrow tuning without keyboard-switching through the
full demo. Press `q` to exit.
"""

import cv2 as cv

from tasks.face_warping import FaceWarping


def main() -> int:
    fw = FaceWarping()
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open default camera.")
        return 1

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = fw.process_frame(frame)
            frame = fw.display_label(frame)
            cv.imshow("FaceWarping Test", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
