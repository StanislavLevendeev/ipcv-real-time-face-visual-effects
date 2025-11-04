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
        self._asset_downscale = 0.33
        self._overlay_cache = {
            "glasses": {"size": None, "rgb": None, "alpha": None},
            "teeth": {"size": None, "rgb": None, "alpha": None},
            "plaster": {"size": None, "rgb": None, "alpha": None},
        }
        self._overlay_quantization = 4  # Round resize targets to multiples of this for cache hits
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
            self._glasses = self._downscale_asset(cv.imread(glasses_path, cv.IMREAD_UNCHANGED))
        
        if os.path.exists(teeth_path):
            self._teeth = self._downscale_asset(cv.imread(teeth_path, cv.IMREAD_UNCHANGED))
        
        if os.path.exists(plaster_path):
            self._plaster = self._downscale_asset(cv.imread(plaster_path, cv.IMREAD_UNCHANGED))

    def _downscale_asset(self, asset):
        """Downscale asset to reduce per-frame resize cost."""
        if asset is None:
            return None
        if asset.shape[0] == 0 or asset.shape[1] == 0:
            return asset
        scale = self._asset_downscale
        if scale >= 1.0:
            return asset
        new_w = max(1, int(asset.shape[1] * scale))
        new_h = max(1, int(asset.shape[0] * scale))
        if new_w == asset.shape[1] and new_h == asset.shape[0]:
            return asset
        return cv.resize(asset, (new_w, new_h), interpolation=cv.INTER_AREA)

    def _quantize_size(self, value):
        """Round size to nearest multiple to improve cache hits."""
        if value <= 0:
            return 0
        step = self._overlay_quantization
        return max(1, int(round(value / step) * step))
    
    def _overlay_png(self, frame, overlay, x, y, w, h, cache_key=None):
        """Overlay a PNG image with alpha channel onto frame"""
        if overlay is None or w <= 0 or h <= 0:
            return frame
        
        # Prefer cached resize if we have one for this size
        cached_rgb = cached_alpha = None
        if cache_key and cache_key in self._overlay_cache:
            cache_entry = self._overlay_cache[cache_key]
            if cache_entry["size"] == (w, h):
                cached_rgb = cache_entry["rgb"]
                cached_alpha = cache_entry["alpha"]
        
        if cached_rgb is None or cached_alpha is None:
            base_h, base_w = overlay.shape[:2]
            if w < base_w or h < base_h:
                interp = cv.INTER_AREA
            else:
                interp = cv.INTER_LINEAR
            overlay_resized = cv.resize(overlay, (w, h), interpolation=interp)
            if overlay_resized.shape[2] == 4:
                alpha = overlay_resized[:, :, 3:4].astype(np.float32) / 255.0
                overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)
            else:
                alpha = np.ones((h, w, 1), dtype=np.float32)
                overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)
            if cache_key and cache_key in self._overlay_cache:
                self._overlay_cache[cache_key]["size"] = (w, h)
                self._overlay_cache[cache_key]["alpha"] = alpha
                self._overlay_cache[cache_key]["rgb"] = overlay_rgb
        else:
            alpha = cached_alpha
            overlay_rgb = cached_rgb
        
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
        overlay_region = overlay_rgb[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        alpha_region = alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        # Extract alpha channel if it exists
        # Blend the overlay with the frame using proper alpha compositing
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = alpha_region * overlay_region + (1.0 - alpha_region) * roi
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        
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
       # frame = self._apply_plaster(frame, nose_bridge, eye_distance)
        
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
        
        plaster_w = self._quantize_size(int(eye_distance * 0.6))
        plaster_h = self._quantize_size(int(eye_distance * 0.6))
        # Position on left glasses lens/frame (not center nose)
        plaster_x = nose_bridge[0] - int(eye_distance * 0.6) - plaster_w // 2
        plaster_y = nose_bridge[1] - plaster_h // 2
        
        return self._overlay_png(frame, self._plaster, plaster_x, plaster_y, plaster_w, plaster_h, cache_key="plaster")
    
    def _apply_glasses(self, frame, left_eye_center, right_eye_center, nose_bridge, eye_distance):
        """Apply glasses aligned with eyes"""
        # Calculate glasses dimensions based on eye distance
        glasses_w = self._quantize_size(int(eye_distance * 2.4))
        glasses_h = self._quantize_size(int(eye_distance * 0.9))
        
        # Position glasses centered between eyes, slightly above
        eyes_center = ((left_eye_center + right_eye_center) / 2).astype(int)
        glasses_x = eyes_center[0] - glasses_w // 2
        glasses_y = eyes_center[1] - int(glasses_h * 0.45)
        
        return self._overlay_png(frame, self._glasses, glasses_x, glasses_y, glasses_w, glasses_h, cache_key="glasses")
    
    def _apply_teeth(self, frame, mouth_top, face_width):
        """Apply bunny teeth at mouth position"""
        teeth_w = self._quantize_size(int(face_width * 0.3))
        teeth_h = self._quantize_size(int(face_width * 0.2))
        teeth_x = mouth_top[0] - teeth_w // 2
        teeth_y = mouth_top[1] - int(teeth_h * 0.2)  # Position higher so top edge is at mouth
        
        return self._overlay_png(frame, self._teeth, teeth_x, teeth_y, teeth_w, teeth_h, cache_key="teeth")

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
