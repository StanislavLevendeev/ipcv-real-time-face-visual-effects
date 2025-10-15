import cv2 as cv
import os

# https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
face_cascade_name = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "haarcascade_frontalface_alt.xml"
    )
)

face_cascade = cv.CascadeClassifier()
face_cascade.load(cv.samples.findFile(face_cascade_name))


def detect_face(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces
