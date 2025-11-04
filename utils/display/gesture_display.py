from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import os

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.labelbottom": False,
        "xtick.bottom": False,
        "ytick.labelleft": False,
        "ytick.left": False,
        "xtick.labeltop": False,
        "xtick.top": False,
        "ytick.labelright": False,
        "ytick.right": False,
    }
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_gesture_on_image(image, results):
    """
    Draws hand landmarks and gesture label on the image.

    Args:
        image: numpy array of the image
        results: GestureRecognizerResult containing gestures and hand_landmarks

    Returns:
        Annotated image with hand landmarks and gesture labels
    """
    if os.getenv("DEBUG", "0") == "0":
        return image

    annotated_image = image.copy()

    # Check if any gestures were detected
    if not results.gestures:
        return annotated_image

    # Draw landmarks for each detected hand
    if results.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Convert landmarks to proto format
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top gesture for this hand
            if hand_idx < len(results.gestures):
                top_gesture = results.gestures[hand_idx][0]  # Get most probable gesture

                # Calculate position for text (near wrist landmark)
                wrist = hand_landmarks[0]
                h, w, _ = annotated_image.shape
                text_x = int(wrist.x * w)
                text_y = int(wrist.y * h) - 20

                # Draw gesture label
                label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                cv.putText(
                    annotated_image,
                    label,
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

    return annotated_image
