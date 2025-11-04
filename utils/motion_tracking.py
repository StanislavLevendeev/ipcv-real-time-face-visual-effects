import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.display.gesture_display import draw_gesture_on_image
import cv2 as cv
import threading
import os
import time

# A global variable to store last result safely
last_gesture_result = None
lock = threading.Lock()

gesture_recognizer_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "gesture_recognizer.task")
)

timestamp = int(time.time() * 1000)


def print_result(result, output_image, timestamp_ms):
    global last_gesture_result
    with lock:
        last_gesture_result = result  # save the latest async result


base_gesture_options = python.BaseOptions(model_asset_path=gesture_recognizer_path)
options_gesture = vision.GestureRecognizerOptions(
    base_options=base_gesture_options,
    num_hands=2,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
)
recognizer_gesture = vision.GestureRecognizer.create_from_options(options_gesture)


def track_gesture(frame):
    global timestamp

    # Update timestamp first
    timestamp = int(time.time() * 1000)

    # Convert to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run async recognition
    recognizer_gesture.recognize_async(image, timestamp)

    # Safely copy latest result
    with lock:
        result_copy = last_gesture_result

    if result_copy:
        rgb_frame = draw_gesture_on_image(rgb_frame, result_copy)

    # Convert back to BGR for OpenCV display
    bgr_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
    return bgr_frame, result_copy


def is_wanted_gesture(result, category):
    if result is None:
        return False
    gestures = []

    for hands in result.gestures:
        gestures.append(hands[0].category_name)
    return category in gestures
