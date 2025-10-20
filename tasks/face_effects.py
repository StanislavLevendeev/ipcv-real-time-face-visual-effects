import cv2 as cv
import numpy as np
import os


class FaceEffects:
    def __init__(self):
        # Load assets
        self._glasses = None
        self._teeth = None
        self._load_assets()
        # Smoothing for jitter reduction
        self._smoothed_faces = []
        self._smoothing_factor = 0.7  # Higher = smoother but more lag
        
    def _load_assets(self):
        """Load PNG assets with alpha channel"""
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        
        glasses_path = os.path.join(assets_dir, "nerdglasses.png")
        teeth_path = os.path.join(assets_dir, "bunnyteeth.png")
        
        if os.path.exists(glasses_path):
            self._glasses = cv.imread(glasses_path, cv.IMREAD_UNCHANGED)
        
        if os.path.exists(teeth_path):
            self._teeth = cv.imread(teeth_path, cv.IMREAD_UNCHANGED)
    
    def _overlay_png(self, frame, overlay, x, y, w, h):
        """Overlay a PNG image with alpha channel onto frame"""
        if overlay is None:
            return frame
        
        # Resize overlay to fit the target area
        overlay_resized = cv.resize(overlay, (w, h))
        
        # Ensure we don't go out of bounds
        if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame
        
        # Extract alpha channel if it exists
        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0
            overlay_rgb = overlay_resized[:, :, :3]
        else:
            alpha = np.ones((h, w))
            overlay_rgb = overlay_resized
        
        # Blend the overlay with the frame
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha * overlay_rgb[:, :, c] + 
                                       (1 - alpha) * frame[y:y+h, x:x+w, c])
        
        return frame
    
    def process_frame(self, frame):
        # Your implementation comes here:
        if not hasattr(self, "_face_cascade"):
            self._face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        if self._face_cascade.empty():
            return frame

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=8, minSize=(100, 100)
        )
        
        # Only process the largest face (most prominent one)
        if len(faces) > 0:
            # Find the largest face by area
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            faces = [largest_face]
        
        # Apply temporal smoothing to reduce jitter
        if len(faces) > 0 and len(self._smoothed_faces) > 0:
            x, y, w, h = faces[0]
            prev_x, prev_y, prev_w, prev_h = self._smoothed_faces[0]
            # Exponential moving average
            x = int(self._smoothing_factor * prev_x + (1 - self._smoothing_factor) * x)
            y = int(self._smoothing_factor * prev_y + (1 - self._smoothing_factor) * y)
            w = int(self._smoothing_factor * prev_w + (1 - self._smoothing_factor) * w)
            h = int(self._smoothing_factor * prev_h + (1 - self._smoothing_factor) * h)
            self._smoothed_faces = [(x, y, w, h)]
            faces = [(x, y, w, h)]
        else:
            self._smoothed_faces = list(faces)

        for (x, y, w, h) in faces:
            # Apply improved freckles FIRST (so they appear behind glasses)
            freckle_color = (89, 47, 20)  # Brown color for freckles
            freckle_positions = [
                # Left cheek - lower and more spread out
                (x + int(0.15 * w), y + int(0.60 * h), 2),
                (x + int(0.20 * w), y + int(0.62 * h), 1),
                (x + int(0.25 * w), y + int(0.58 * h), 2),
                (x + int(0.18 * w), y + int(0.65 * h), 1),
                (x + int(0.28 * w), y + int(0.63 * h), 2),
                (x + int(0.23 * w), y + int(0.67 * h), 1),
                (x + int(0.16 * w), y + int(0.68 * h), 2),
                (x + int(0.22 * w), y + int(0.70 * h), 1),
                (x + int(0.27 * w), y + int(0.69 * h), 2),
                (x + int(0.19 * w), y + int(0.72 * h), 1),
                # Right cheek - lower and more spread out
                (x + int(0.85 * w), y + int(0.60 * h), 2),
                (x + int(0.80 * w), y + int(0.62 * h), 1),
                (x + int(0.75 * w), y + int(0.58 * h), 2),
                (x + int(0.82 * w), y + int(0.65 * h), 1),
                (x + int(0.72 * w), y + int(0.63 * h), 2),
                (x + int(0.77 * w), y + int(0.67 * h), 1),
                (x + int(0.84 * w), y + int(0.68 * h), 2),
                (x + int(0.78 * w), y + int(0.70 * h), 1),
                (x + int(0.73 * w), y + int(0.69 * h), 2),
                (x + int(0.81 * w), y + int(0.72 * h), 1),
                # Nose bridge - below glasses
                (x + int(0.48 * w), y + int(0.58 * h), 1),
                (x + int(0.52 * w), y + int(0.59 * h), 1),
                (x + int(0.50 * w), y + int(0.62 * h), 1),
            ]
            
            for (fx, fy, radius) in freckle_positions:
                cv.circle(frame, (fx, fy), radius, freckle_color, -1)
            
            # Apply nerd glasses AFTER freckles (so they appear on top)
            glasses_y = y + int(0.25 * h)
            glasses_w = int(1.1 * w)
            glasses_h = int(0.4 * h)
            glasses_x = x - int(0.05 * w)
            frame = self._overlay_png(frame, self._glasses, glasses_x, glasses_y, glasses_w, glasses_h)
            
            # Apply bunny teeth - positioned so top aligns with mouth and extends downward
            teeth_x = x + int(0.35 * w)
            teeth_y = y + int(0.72 * h)  # Start slightly above mouth level
            teeth_w = int(0.30 * w)
            teeth_h = int(0.25 * h)  # Extend downward from mouth
            frame = self._overlay_png(frame, self._teeth, teeth_x, teeth_y, teeth_w, teeth_h)
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
