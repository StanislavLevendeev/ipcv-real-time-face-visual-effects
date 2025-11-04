import cv2 as cv
import mediapipe as mp
import numpy as np
import pygame
import os

class FaceWarping:
    """
    Chad Jaw Filter: Applies aggressive facial transformations
    Features: jaw widening, beard overlay, angry eyebrows, dark aesthetic, flash effects
    """
    
    # CONFIGURATION PARAMETERS
    
    # Audio Settings
    ENABLE_AUDIO = True
    VOLUME = 0.2
    MUSIC_FILE = None  # Will be set dynamically
    
    # Visual Effects
    SATURATION = 2.0 # Color intensity (1.0 = normal, >2.0 = very vivid)
    BRIGHTNESS = 1.1 # Image darkness (0.0 = black, 1.0 = normal)
    CONTRAST = 1.3 # Contrast boost (1.0 = normal, >1.0 = more contrast)
    
    # Flash Effect
    ENABLE_FLASH = True
    FLASH_INTERVAL = 10 # Frames between flashes (~0.7s at 30fps) - MORE FREQUENT
    FLASH_DURATION = 3 # Flash length in frames
    FLASH_INTENSITY = 1 # Flash brightness (0.0-1.0)
    
    # Eyebrow Effect (Angry Look) 
    EYEBROW_VERTICAL_SHIFT_RATIO = 0.055     
    EYEBROW_HORIZONTAL_SHIFT_RATIO = 0.038   
    EYEBROW_OUTER_LIFT_RATIO = 0.020         
    EYEBROW_INFLUENCE_RATIO = 0.050          
    EYEBROW_ROI_MARGIN_RATIO = 0.125         
    
    # Jaw Warping 
    JAW_WIDTH_SCALE = 1.35 # Horizontal expansion (1.0 = no change)
    CHIN_EXTENSION_RATIO = 0.013     
    JAW_INFLUENCE_RATIO = 0.055      
    CHIN_INFLUENCE_RATIO = 0.030     
    JAW_ROI_MARGIN_RATIO = 0.200     
    
    # Beard Effect 
    BEARD_CHEEK_RADIUS_RATIO = 0.150         
    BEARD_CHEEK_HEIGHT_OFFSET_RATIO = 0.075  
    BEARD_CHIN_WIDTH_RATIO = 0.175           
    BEARD_CHIN_LENGTH_RATIO = 0.150          
    BEARD_BLUR_SIZE_RATIO = 0.228             
    BEARD_BLUR_SIGMA_RATIO = 0.113           
    BEARD_DARKNESS = 0.45
    BEARD_TINT_BGR = (20, 30, 50) # BGR color
    BEARD_GRADIENT_START_RATIO = 0.300       
    BEARD_GRADIENT_RANGE_RATIO = 0.350       
    
    # MediaPipe Face Mesh Indices
    JAW_INDICES = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                   377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    LOWER_JAW_INDICES = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
    CHIN_INDEX = 152
    LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300]
    
    def __init__(self, debug_mode=False):
        """Initialize face mesh, audio, and visual effects"""
        self.debug_mode = debug_mode
        
        # Set correct music path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.MUSIC_FILE = os.path.join(base_dir, "..", "music", "chad_music.wav")
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Audio state
        self.audio_initialized = False
        self.music_playing = False
        self._initialize_audio()
        
        # Flash state
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    def _initialize_audio(self):
        """Initialize pygame audio system"""
        if not self.ENABLE_AUDIO:
            return
        
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            if os.path.exists(self.MUSIC_FILE):
                pygame.mixer.music.load(self.MUSIC_FILE)
                pygame.mixer.music.set_volume(self.VOLUME)
                self.audio_initialized = True
            else:
                print(f"[FaceWarping] Audio file '{self.MUSIC_FILE}' not found")
        except Exception as e:
            print(f"[FaceWarping] Audio initialization failed: {e}")

    def _start_music(self):
        """Start background music"""
        if self.audio_initialized and not self.music_playing:
            try:
                pygame.mixer.music.play(-1)
                self.music_playing = True
            except:
                pass

    def _stop_music(self):
        """Stop background music"""
        if self.music_playing:
            try:
                pygame.mixer.music.stop()
                self.music_playing = False
            except:
                pass

    def process_frame(self, frame):
        """Main processing pipeline"""
        self._start_music()
        
        h, w, _ = frame.shape
        
        # Apply color grading before grayscale conversion
        frame = self._apply_color_effects(frame)
        
        # Convert to black and white
        frame = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        
        # Detect face landmarks
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return self._apply_flash_effect(frame)

        # Extract and process facial landmarks
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        
        jaw_pts = points[self.JAW_INDICES]
        lower_jaw_pts = points[self.LOWER_JAW_INDICES]
        chin_pt = points[self.CHIN_INDEX]
        left_brow = points[self.LEFT_EYEBROW_INDICES]
        right_brow = points[self.RIGHT_EYEBROW_INDICES]

        # Calculate face width for proportional scaling
        face_width = np.max(jaw_pts[:, 0]) - np.min(jaw_pts[:, 0])

        if self.debug_mode:
            frame = self._draw_debug_info(frame, jaw_pts, lower_jaw_pts, chin_pt)
        else:
            frame = self._apply_jaw_warp(frame, lower_jaw_pts, chin_pt, face_width)
            frame = self._add_beard_effect(frame, jaw_pts, lower_jaw_pts, chin_pt, face_width)
            frame = self._apply_eyebrow_frown(frame, left_brow, right_brow, face_width)

        return self._apply_flash_effect(frame)

    def stop(self):
        """Called when switching away from this task - stops music"""
        self._stop_music()
        # Reset flash state
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    def _apply_color_effects(self, frame):
        """Apply saturation, brightness, and contrast"""
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.SATURATION, 0, 255)
        
        frame = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR).astype(np.float32)
        frame = self.CONTRAST * (frame * self.BRIGHTNESS - 128) + 128
        
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _apply_flash_effect(self, frame):
        """Apply periodic white flash effect"""
        if not self.ENABLE_FLASH:
            return frame
        
        self.frame_count += 1
        
        if self.frame_count % self.FLASH_INTERVAL == 0:
            self.flash_active = True
            self.flash_timer = self.FLASH_DURATION
        
        if self.flash_active:
            fade = (self.flash_timer / self.FLASH_DURATION) * self.FLASH_INTENSITY
            white = np.full_like(frame, 255)
            frame = cv.addWeighted(frame, 1 - fade, white, fade, 0)
            
            self.flash_timer -= 1
            if self.flash_timer <= 0:
                self.flash_active = False
        
        return frame

    def _apply_eyebrow_frown(self, frame, left_brow, right_brow, face_width):
        """Apply aggressive eyebrow transformation for intense angry expression"""
        h, w = frame.shape[:2]
        
        # Calculate proportional parameters based on face size
        v_shift = int(face_width * self.EYEBROW_VERTICAL_SHIFT_RATIO)
        h_shift = int(face_width * self.EYEBROW_HORIZONTAL_SHIFT_RATIO)
        outer_lift = int(face_width * self.EYEBROW_OUTER_LIFT_RATIO)
        influence_radius = int(face_width * self.EYEBROW_INFLUENCE_RATIO)
        roi_margin = int(face_width * self.EYEBROW_ROI_MARGIN_RATIO)
        
        new_left = left_brow.copy().astype(np.float32)
        new_right = right_brow.copy().astype(np.float32)

        # Create intense frown by moving eyebrows down and inward
        for i in range(len(left_brow)):
            factor = (len(left_brow) - i) / len(left_brow)
            
            # Inner points move down and toward center (angry V-shape)
            new_left[i, 1] += v_shift * factor
            new_left[i, 0] += h_shift * factor
            new_right[i, 1] += v_shift * factor
            new_right[i, 0] -= h_shift * factor
            
            # Outer points lift slightly for dramatic arch
            if i > 2:
                new_left[i, 1] -= outer_lift
                new_right[i, 1] -= outer_lift

        # Define ROI and create displacement maps
        x, y, bw, bh = cv.boundingRect(np.vstack([left_brow, right_brow]).astype(np.int32))
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)
        
        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x,
            np.arange(bh, dtype=np.float32) + y
        )

        # Apply Gaussian-weighted displacement
        radius_sq = 2 * influence_radius ** 2
        
        for src, dst in zip(np.vstack([left_brow, right_brow]), 
                           np.vstack([new_left, new_right])):
            dx, dy = dst - src
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2
            influence = np.exp(-dist_sq / radius_sq)
            map_x -= dx * influence
            map_y -= dy * influence

        # Apply remapping
        roi = frame[y:y+bh, x:x+bw]
        if roi.size > 0:
            frame[y:y+bh, x:x+bw] = cv.remap(
                roi, map_x - x, map_y - y, 
                cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
        
        return frame

    def _apply_jaw_warp(self, frame, lower_jaw_pts, chin_pt, face_width):
        """Apply jaw widening and chin extension"""
        h, w = frame.shape[:2]
        
        # Calculate proportional parameters based on face size
        chin_ext = int(face_width * self.CHIN_EXTENSION_RATIO)
        jaw_influence = face_width * self.JAW_INFLUENCE_RATIO
        chin_influence = face_width * self.CHIN_INFLUENCE_RATIO
        roi_margin = int(face_width * self.JAW_ROI_MARGIN_RATIO)
        
        # Calculate transformations
        center = np.mean(lower_jaw_pts, axis=0)
        new_jaw = ((lower_jaw_pts - center) * [self.JAW_WIDTH_SCALE, 1.0] + center).astype(np.float32)
        chin_new = chin_pt.astype(np.float32) + [0, chin_ext]

        # Define ROI
        all_pts = np.vstack([lower_jaw_pts, [chin_pt]])
        x, y, bw, bh = cv.boundingRect(all_pts.astype(np.int32))
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)

        # Create displacement maps
        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x,
            np.arange(bh, dtype=np.float32) + y
        )

        # Apply jaw widening
        for src, dst in zip(lower_jaw_pts, new_jaw):
            dx, dy = dst - src
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2
            influence = np.exp(-dist_sq / (2 * jaw_influence ** 2))
            map_x -= dx * influence
            map_y -= dy * influence

        # Apply chin extension
        dx_chin, dy_chin = chin_new - chin_pt
        dist_sq_chin = (map_x - chin_pt[0]) ** 2 + (map_y - chin_pt[1]) ** 2
        influence_chin = np.exp(-dist_sq_chin / (2 * chin_influence ** 2))
        map_x -= dx_chin * influence_chin
        map_y -= dy_chin * influence_chin

        # Apply warping
        roi = frame[y:y+bh, x:x+bw]
        if roi.size > 0:
            frame[y:y+bh, x:x+bw] = cv.remap(
                roi, map_x - x, map_y - y,
                cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )

        return frame

    def _add_beard_effect(self, frame, jaw_pts, lower_jaw_pts, chin_pt, face_width):
        """Apply full beard overlay on jaw, chin, and cheeks"""
        h, w = frame.shape[:2]
        
        # Calculate proportional parameters based on face size
        cheek_radius = int(face_width * self.BEARD_CHEEK_RADIUS_RATIO)
        cheek_offset = int(face_width * self.BEARD_CHEEK_HEIGHT_OFFSET_RATIO)
        chin_width = int(face_width * self.BEARD_CHIN_WIDTH_RATIO)
        chin_length = int(face_width * self.BEARD_CHIN_LENGTH_RATIO)
        blur_size = int(face_width * self.BEARD_BLUR_SIZE_RATIO)
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Must be odd
        blur_sigma = face_width * self.BEARD_BLUR_SIGMA_RATIO
        gradient_start = int(face_width * self.BEARD_GRADIENT_START_RATIO)
        gradient_range = int(face_width * self.BEARD_GRADIENT_RANGE_RATIO)
        
        mask = np.zeros((h, w), dtype=np.float32)

        # Fill jaw contour
        cv.fillConvexPoly(mask, cv.convexHull(jaw_pts), 1.0)

        # Extend onto cheeks
        for pt in jaw_pts[::2]:
            cheek_pt = (pt[0], max(0, pt[1] - cheek_offset))
            cv.circle(mask, tuple(cheek_pt), cheek_radius, 1.0, -1)
        
        for pt in lower_jaw_pts[::2]:
            cv.circle(mask, tuple(pt), cheek_radius, 1.0, -1)

        # Extend below chin
        chin_ext = np.array([
            [chin_pt[0] - chin_width, chin_pt[1]],
            [chin_pt[0] + chin_width, chin_pt[1]],
            [chin_pt[0] + chin_width + 10, chin_pt[1] + chin_length],
            [chin_pt[0] - chin_width - 10, chin_pt[1] + chin_length]
        ], dtype=np.int32)
        cv.fillConvexPoly(mask, chin_ext, 1.0)

        # Smooth edges
        mask = cv.GaussianBlur(mask, (blur_size, blur_size), blur_sigma)

        # Apply vertical gradient
        y_coords = np.arange(h).reshape(-1, 1)
        gradient = np.clip((y_coords - chin_pt[1] + gradient_start) / gradient_range, 0, 1)
        mask *= gradient

        # Create darkened beard overlay
        beard = (frame * self.BEARD_DARKNESS).astype(np.uint8)
        tint = np.full_like(frame, self.BEARD_TINT_BGR)
        beard = cv.addWeighted(beard, 0.8, tint, 0.2, 0)

        # Blend with frame
        mask_3ch = np.stack([mask] * 3, axis=-1)
        return ((frame * (1 - mask_3ch) + beard * mask_3ch)).astype(np.uint8)

    def _draw_debug_info(self, frame, jaw_pts, lower_jaw_pts, chin_pt):
        """Draw debug visualization"""
        center = np.mean(lower_jaw_pts, axis=0).astype(np.int32)
        new_jaw = ((lower_jaw_pts - center) * [1.4, 1.0] + center).astype(np.int32)
        chin_new = chin_pt + [0, 8]

        for pt in lower_jaw_pts:
            cv.circle(frame, tuple(pt), 3, (255, 0, 0), -1)
        for pt in new_jaw:
            cv.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
        
        cv.circle(frame, tuple(chin_pt), 5, (0, 255, 0), -1)
        cv.circle(frame, tuple(chin_new), 5, (0, 255, 255), -1)

        cv.polylines(frame, [cv.convexHull(lower_jaw_pts)], True, (255, 0, 0), 2)
        cv.polylines(frame, [cv.convexHull(new_jaw)], True, (0, 0, 255), 2)

        return frame
    
    def display_label(self, frame):
        """Display mode label"""
        mode = "DEBUG MODE" if self.debug_mode else "WARP MODE"
        cv.putText(frame, f"Chad Jaw Filter - {mode}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        if self.debug_mode:
            cv.putText(frame, "Blue: Original | Red: Target | Yellow: Vectors", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        return frame
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_music()