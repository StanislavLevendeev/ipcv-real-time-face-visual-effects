import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.display.hand_display import draw_landmarks_on_image
from utils.display.gesture_display import draw_gesture_on_image
import threading
import os

# A global variable to store last result safely
last_gesture_result = None
lock = threading.Lock()

hand_marker_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "hand_landmarker.task"
    )
)

gesture_recognizer_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "gesture_recognizer.task"
    )
)

base_hand_options = python.BaseOptions(model_asset_path=hand_marker_path)
hand_options = vision.HandLandmarkerOptions(base_options=base_hand_options, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

timestamp = 0


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


def track_hand(frame):
    # STEP 4: Detect hand landmarks from the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = hand_detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    return annotated_image


def track_gesture(frame):
    global timestamp
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognizer_gesture.recognize_async(image, timestamp)

    timestamp += 33  # approximate 30 FPS

    gestures = []
    with lock:
        if last_gesture_result:
            gestures = []
            for hands in last_gesture_result.gestures:
                gestures.append(hands[0].category_name)
            frame = draw_gesture_on_image(frame, last_gesture_result)
    return frame, last_gesture_result


def is_wanted_gesture(result, category):
    if result is None:
        return False
    gestures = []

    for hands in result.gestures:
        gestures.append(hands[0].category_name)
    return category in gestures
