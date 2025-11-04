import cv2 as cv
import numpy as np
import os
import mediapipe as mp


class FaceEffects:
    # MediaPipe Face Mesh Landmark Indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Left eye outline
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Right eye outline
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    MOUTH_TOP = 13  # Upper lip center
    MOUTH_BOTTOM = 14  # Lower lip center
    
    # Cheek landmarks for freckles (multiple points for better coverage)
    LEFT_CHEEK_LANDMARKS = [205, 206, 203, 50, 119, 118, 117, 116, 111, 100]
    RIGHT_CHEEK_LANDMARKS = [425, 426, 423, 280, 348, 347, 346, 345, 340, 329]
    NOSE_SIDE_LANDMARKS = [114, 188, 122, 245, 412, 343]  # Sides of nose
    
    def __init__(self):
        # Load assets
        self._glasses = None
        self._teeth = None
        self._plaster = None
        self._load_assets()
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Smoothing for jitter reduction
        self._smoothed_landmarks = None
        self._smoothing_factor = 0.6  # Higher = smoother but more lag
        
    def _load_assets(self):
        """Load PNG assets with alpha channel"""
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        
        glasses_path = os.path.join(assets_dir, "nerdglasses.png")
        teeth_path = os.path.join(assets_dir, "bunnyteeth.png")
        plaster_path = os.path.join(assets_dir, "plaster.png")
        
        if os.path.exists(glasses_path):
            self._glasses = cv.imread(glasses_path, cv.IMREAD_UNCHANGED)
        
        if os.path.exists(teeth_path):
            self._teeth = cv.imread(teeth_path, cv.IMREAD_UNCHANGED)
        
        if os.path.exists(plaster_path):
            self._plaster = cv.imread(plaster_path, cv.IMREAD_UNCHANGED)
    
    def _overlay_png(self, frame, overlay, x, y, w, h):
        """Overlay a PNG image with alpha channel onto frame"""
        if overlay is None or w <= 0 or h <= 0:
            return frame
        
        # Resize overlay to fit the target area
        overlay_resized = cv.resize(overlay, (w, h), interpolation=cv.INTER_AREA)
        
        frame_h, frame_w = frame.shape[:2]
        # Clip overlay to stay inside the frame instead of discarding it completely
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, frame_w)
        y2 = min(y + h, frame_h)
        if x1 >= x2 or y1 >= y2:
            return frame
        
        overlay_x1 = x1 - x
        overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)
        overlay_region = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        # Extract alpha channel if it exists
        if overlay_region.shape[2] == 4:
            alpha = overlay_region[:, :, 3:4] / 255.0
            overlay_rgb = overlay_region[:, :, :3]
        else:
            alpha = np.ones((overlay_region.shape[0], overlay_region.shape[1], 1), dtype=np.float32)
            overlay_rgb = overlay_region
        
        # Blend the overlay with the frame using proper alpha compositing
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        overlay_rgb = overlay_rgb.astype(np.float32)
        blended = alpha * overlay_rgb + (1.0 - alpha) * roi
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return frame
    
    def _apply_round_face_warp(self, frame, points, strength=0.13):
        h, w = frame.shape[:2]
        # Use jawline (0-16) and outer cheeks (234, 454) in proper order for closed contour
        jaw_indices = list(range(0, 17))
        left_cheek = 234
        right_cheek = 454
        contour_indices = jaw_indices + [right_cheek] + [left_cheek]
        contour_pts = points[contour_indices]
        center = np.mean(contour_pts, axis=0).astype(np.int32)
        
        # Create mask for the face contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.fillConvexPoly(mask, contour_pts.astype(np.int32), 255)
        
        # Get bounding rect for efficiency
        x, y, bw, bh = cv.boundingRect(contour_pts.astype(np.int32))
        x = max(0, x - 40)
        y = max(0, y - 40)
        bw = min(w - x, bw + 80)
        bh = min(h - y, bh + 80)
        
        # Create coordinate grids in ROI space (0 to bw/bh)
        map_x = np.arange(bw, dtype=np.float32)[np.newaxis, :].repeat(bh, axis=0)
        map_y = np.arange(bh, dtype=np.float32)[:, np.newaxis].repeat(bw, axis=1)
        
        # Get ROI
        roi = frame[y:y+bh, x:x+bw]
        
        # Only morph pixels near the contour
        for pt in contour_pts:
            dx = (pt[0] - center[0]) * strength
            dy = (pt[1] - center[1]) * strength
            # Calculate distance in frame coordinates
            pt_x_frame = pt[0]
            pt_y_frame = pt[1]
            # Convert map coordinates to frame coordinates for distance calculation
            map_x_frame = map_x + x
            map_y_frame = map_y + y
            dist_sq = (map_x_frame - pt_x_frame) ** 2 + (map_y_frame - pt_y_frame) ** 2
            influence = np.exp(-dist_sq / (2 * (bw * 0.18) ** 2))
            map_x += dx * influence
            map_y += dy * influence
        
        # Apply the warp
        warped = cv.remap(roi, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        
        # Blend using mask for smooth edges
        mask_roi = mask[y:y+bh, x:x+bw]
        mask_blur = cv.GaussianBlur(mask_roi, (51, 51), 0) / 255.0
        frame[y:y+bh, x:x+bw] = (warped * mask_blur[..., None] + roi * (1 - mask_blur[..., None])).astype(np.uint8)
        
        return frame

    def process_frame(self, frame):
        """Process frame with MediaPipe face mesh for precise landmark detection"""
        h, w, _ = frame.shape
        
        # Increase saturation for more vivid, cartoonish look
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # Boost saturation by 40%
        frame = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
        
        # Convert to RGB for MediaPipe
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return frame
        # Extract landmarks as pixel coordinates
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        
        # Apply temporal smoothing to reduce jitter
        if self._smoothed_landmarks is not None:
            points = (self._smoothing_factor * self._smoothed_landmarks + 
                     (1 - self._smoothing_factor) * points).astype(np.int32)
        self._smoothed_landmarks = points.astype(np.float32)
        
        # Extract key facial features
        left_eye = points[self.LEFT_EYE_INDICES]
        right_eye = points[self.RIGHT_EYE_INDICES]
        nose_tip = points[self.NOSE_TIP]
        nose_bridge = points[self.NOSE_BRIDGE]
        mouth_top = points[self.MOUTH_TOP]
        left_cheek_points = points[self.LEFT_CHEEK_LANDMARKS]
        right_cheek_points = points[self.RIGHT_CHEEK_LANDMARKS]
        nose_sides = points[self.NOSE_SIDE_LANDMARKS]
        
        # Calculate eye centers and face width
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        face_width = eye_distance * 2.8  # Approximate face width from eye distance
        
        # Apply freckles FIRST (behind glasses)
        frame = self._apply_freckles(frame, left_cheek_points, right_cheek_points, nose_sides, face_width)
        
        # Apply glasses (on top of freckles)
        frame = self._apply_glasses(frame, left_eye_center, right_eye_center, nose_bridge, eye_distance)
        
        # Add plaster on glasses bridge
        frame = self._apply_plaster(frame, nose_bridge, eye_distance)
        
        # Apply teeth (at mouth position)
        frame = self._apply_teeth(frame, mouth_top, face_width)
        
        return frame
    
    def _apply_freckles(self, frame, left_cheek_points, right_cheek_points, nose_sides, face_width):
        """Apply freckles based on precise cheek landmarks with natural randomness"""
        freckle_color = (40, 100, 150)  # True brown in BGR for OpenCV
        # Use a fixed seed based on landmark positions for consistent freckle placement
        np.random.seed(42)
        freckle_positions = []
        # Left cheek freckles - scatter around actual cheek landmarks
        if left_cheek_points is not None:
            for point in left_cheek_points:
                num_freckles = np.random.randint(2, 4)
                for _ in range(num_freckles):
                    offset_x = np.random.randint(-15, 15)
                    offset_y = np.random.randint(-15, 15)
                    x = point[0] + offset_x
                    y = point[1] + offset_y
                    radius = np.random.choice([2, 3], p=[0.5, 0.5])
                    freckle_positions.append((x, y, radius))
        # Right cheek freckles - scatter around actual cheek landmarks
        if right_cheek_points is not None:
            for point in right_cheek_points:
                num_freckles = np.random.randint(2, 4)
                for _ in range(num_freckles):
                    offset_x = np.random.randint(-15, 15)
                    offset_y = np.random.randint(-15, 15)
                    x = point[0] + offset_x
                    y = point[1] + offset_y
                    radius = np.random.choice([2, 3], p=[0.5, 0.5])
                    freckle_positions.append((x, y, radius))
        # Nose freckles - few small ones on nose sides
        if nose_sides is not None:
            for point in nose_sides:
                if np.random.random() > 0.5:
                    offset_x = np.random.randint(-5, 5)
                    offset_y = np.random.randint(-5, 5)
                    x = point[0] + offset_x
                    y = point[1] + offset_y
                    freckle_positions.append((x, y, 2))
        # Reset seed for normal operation
        np.random.seed(None)
        # Draw all freckles with slight transparency to blend better
        if frame is not None:
            # Create overlay for freckles
            overlay = frame.copy()
            for (fx, fy, radius) in freckle_positions:
                cv.circle(overlay, (int(fx), int(fy)), radius, freckle_color, -1)
            # Blend with slight transparency
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame
    
    def _apply_plaster(self, frame, nose_bridge, eye_distance):
        """Add plaster/bandage on glasses bridge"""
        if self._plaster is None:
            return frame
        
        plaster_w = int(eye_distance * 0.6)
        plaster_h = int(eye_distance * 0.6)
        # Position on left glasses lens/frame (not center nose)
        plaster_x = nose_bridge[0] - int(eye_distance * 0.6) - plaster_w // 2
        plaster_y = nose_bridge[1] - plaster_h // 2
        
        return self._overlay_png(frame, self._plaster, plaster_x, plaster_y, plaster_w, plaster_h)
    
    def _apply_glasses(self, frame, left_eye_center, right_eye_center, nose_bridge, eye_distance):
        """Apply glasses aligned with eyes"""
        # Calculate glasses dimensions based on eye distance
        glasses_w = int(eye_distance * 2.4)
        glasses_h = int(eye_distance * 0.9)
        
        # Position glasses centered between eyes, slightly above
        eyes_center = ((left_eye_center + right_eye_center) / 2).astype(int)
        glasses_x = eyes_center[0] - glasses_w // 2
        glasses_y = eyes_center[1] - int(glasses_h * 0.45)
        
        return self._overlay_png(frame, self._glasses, glasses_x, glasses_y, glasses_w, glasses_h)
    
    def _apply_teeth(self, frame, mouth_top, face_width):
        """Apply bunny teeth at mouth position"""
        teeth_w = int(face_width * 0.3)
        teeth_h = int(face_width * 0.2)
        teeth_x = mouth_top[0] - teeth_w // 2
        teeth_y = mouth_top[1] - int(teeth_h * 0.2)  # Position higher so top edge is at mouth
        
        return self._overlay_png(frame, self._teeth, teeth_x, teeth_y, teeth_w, teeth_h)

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
