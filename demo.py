import cv2 as cv
from tasks.face_effects import FaceEffects
from tasks.face_warping import FaceWarping
from tasks.motion_tracking import MotionTracking
from tasks.combined_task import CombinedTask
from tasks.task_manager import TaskManager


tasks = [FaceEffects(), FaceWarping(), MotionTracking(), CombinedTask()]
task_manager = TaskManager(tasks)

cam = cv.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    key = cv.waitKey(1)
    # Process frame with selected task
    processed_frame = task_manager.process_frame(frame, key)

    cv.imshow("Camera", processed_frame)

    # Press 'q' to exit the loop
    if key == ord("q"):
        break

cam.release()
cv.destroyAllWindows()
