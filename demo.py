"""Real-time face visual effects demo.

Streams the webcam through a selectable pipeline step (face effects, face
warping, motion tracking, combined). Number keys 1-4 switch tasks, `d` toggles
debug overlays, `q` quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
from dotenv import load_dotenv

from tasks.combined_task import CombinedTask
from tasks.face_effects import FaceEffects
from tasks.face_warping import FaceWarping
from tasks.motion_tracking import MotionTracking
from tasks.task_manager import TaskManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable the horizontal mirror (default: mirror on, like a selfie cam).",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(".env"))

    tasks = [FaceEffects(), FaceWarping(), MotionTracking(), CombinedTask()]
    manager = TaskManager(tasks)

    cam = cv.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f"Could not open camera index {args.camera}.")
        return 1

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            if not args.no_flip:
                frame = cv.flip(frame, 1)

            key = cv.waitKey(1) & 0xFF
            frame = manager.process_frame(frame, key)
            cv.imshow("Camera", frame)

            if key == ord("q"):
                break
    finally:
        cam.release()
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
