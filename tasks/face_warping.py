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
    
    # Visual Effects
    SATURATION = 2.0 # Color intensity (1.0 = normal, >2.0 = very vivid)
    BRIGHTNESS = 1.1 # Image darkness (0.0 = black, 1.0 = normal)
    CONTRAST = 1.3 # Contrast boost (1.0 = normal, >1.0 = more contrast)
    
    # Flash Effect
    ENABLE_FLASH = True
    FLASH_INTERVAL = 10 
    FLASH_DURATION = 3
    FLASH_INTENSITY = 1 
    
    # Eyebrow Effect (Angry Look) 
    EYEBROW_VERTICAL_SHIFT_RATIO = 0.055     
    EYEBROW_HORIZONTAL_SHIFT_RATIO = 0.038   
    EYEBROW_OUTER_LIFT_RATIO = 0.020         
    EYEBROW_INFLUENCE_RATIO = 0.050          
    EYEBROW_ROI_MARGIN_RATIO = 0.125         
    
    # Jaw Warping 
    JAW_WIDTH_SCALE = 1.35 
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
    BEARD_TINT_BGR = (20, 30, 50)
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
        """Initialize face mesh and visual effects"""
        self.debug_mode = debug_mode
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Flash state
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    def process_frame(self, frame):
        """Main processing pipeline"""        
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
        # Convert normalized (0–1) landmark coordinates from mediapipe into actual pixel positions
        # by scaling with frame width (w) and height (h), then store them as array.
        points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        
        jaw_pts = points[self.JAW_INDICES]
        lower_jaw_pts = points[self.LOWER_JAW_INDICES]
        chin_pt = points[self.CHIN_INDEX]
        left_brow = points[self.LEFT_EYEBROW_INDICES]
        right_brow = points[self.RIGHT_EYEBROW_INDICES]

        # Calculate face width for proportional scaling
        face_width = np.max(jaw_pts[:, 0]) - np.min(jaw_pts[:, 0])

        if os.environ.get("DEBUG", "0") == "1":
            frame = self._draw_debug_info(frame, jaw_pts, lower_jaw_pts, chin_pt)
        
        frame = self._apply_jaw_warp(frame, lower_jaw_pts, chin_pt, face_width)
        frame = self._add_beard_effect(frame, jaw_pts, lower_jaw_pts, chin_pt, face_width)
        frame = self._apply_eyebrow_frown(frame, left_brow, right_brow, face_width)

        return self._apply_flash_effect(frame)

    def stop(self):
        """Called when switching away from this task"""
        # Reset flash state
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    def _apply_color_effects(self, frame):
        """Enhance frame with saturation, brightness, and contrast adjustments."""
        # Convert BGR image to HSV to easily modify saturation
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        
        # Boost or reduce color saturation, then ensure values stay in valid range [0, 255]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.SATURATION, 0, 255)
        
        # Convert back to BGR
        frame = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR).astype(np.float32)
        
        # Apply brightness and contrast formula:
        # new_pixel = contrast * (brightness * pixel - 128) + 128
        frame = self.CONTRAST * (frame * self.BRIGHTNESS - 128) + 128
        
        # Clip values to [0, 255] and convert back to uint8 for OpenCV compatibility
        return np.clip(frame, 0, 255).astype(np.uint8)


    def _apply_flash_effect(self, frame):
        """Add a short white flash effect every few frames."""
        if not self.ENABLE_FLASH:
            return frame

        # Count frames to control when to trigger the flash
        self.frame_count += 1

        # Activate flash periodically based on predefined interval
        if self.frame_count % self.FLASH_INTERVAL == 0:
            self.flash_active = True
            self.flash_timer = self.FLASH_DURATION

        if self.flash_active:
            # Compute current flash intensity (fades over time)
            fade = (self.flash_timer / self.FLASH_DURATION) * self.FLASH_INTENSITY
            
            # Blend current frame with white image to create flash effect
            white = np.full_like(frame, 255)
            frame = cv.addWeighted(frame, 1 - fade, white, fade, 0)
            
            # Decrease flash timer; stop flash when duration is over
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

        # Define the region of interest (ROI) around both eyebrows, adding a margin for smooth warping
        x, y, bw, bh = cv.boundingRect(np.vstack([left_brow, right_brow]).astype(np.int32))
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)

        # Create meshgrid of pixel coordinates within the ROI for remapping
        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x,
            np.arange(bh, dtype=np.float32) + y
        )

        # Apply Gaussian-weighted displacement from original to target eyebrow positions
        # Pixels closer to a landmark move more; influence decreases with distance squared
        radius_sq = 2 * influence_radius ** 2
        for src, dst in zip(np.vstack([left_brow, right_brow]), np.vstack([new_left, new_right])):
            dx, dy = dst - src  # displacement vector
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2  # squared distance to landmark
            influence = np.exp(-dist_sq / radius_sq)  # Gaussian falloff
            map_x -= dx * influence  # apply horizontal displacement
            map_y -= dy * influence  # apply vertical displacement

        # Extract ROI from frame and apply the remapping to warp eyebrows
        roi = frame[y:y+bh, x:x+bw]
        if roi.size > 0:
            frame[y:y+bh, x:x+bw] = cv.remap(
                roi, map_x - x, map_y - y, 
                cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )

        return frame


    def _apply_jaw_warp(self, frame, lower_jaw_pts, chin_pt, face_width):
        """Apply jaw widening and chin extension to create exaggerated facial effects."""
        
        h, w = frame.shape[:2]  # Get frame dimensions
        
        # Compute proportional parameters based on face width
        chin_ext = int(face_width * self.CHIN_EXTENSION_RATIO)       # How much to extend the chin downward
        jaw_influence = face_width * self.JAW_INFLUENCE_RATIO       # Spread of jaw widening effect
        chin_influence = face_width * self.CHIN_INFLUENCE_RATIO     # Spread of chin extension effect
        roi_margin = int(face_width * self.JAW_ROI_MARGIN_RATIO)    # Extra padding around jaw/chin for smooth warping
        
        # Calculate new target positions for jaw and chin
        center = np.mean(lower_jaw_pts, axis=0)  # Compute jaw center
        # Widen jaw horizontally by scaling x-coordinates while keeping y the same
        new_jaw = ((lower_jaw_pts - center) * [self.JAW_WIDTH_SCALE, 1.0] + center).astype(np.float32)
        # Move chin downward
        chin_new = chin_pt.astype(np.float32) + [0, chin_ext]

        # Define the region of interest (ROI) around jaw and chin, adding margin
        all_pts = np.vstack([lower_jaw_pts, [chin_pt]])
        x, y, bw, bh = cv.boundingRect(all_pts.astype(np.int32))
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)

        # Create meshgrid of pixel coordinates within ROI for remapping
        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x,
            np.arange(bh, dtype=np.float32) + y
        )

        # Apply jaw widening: pixels near jaw landmarks move according to a Gaussian falloff
        for src, dst in zip(lower_jaw_pts, new_jaw):
            dx, dy = dst - src                  # Displacement vector for each landmark
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2  # Squared distance to pixel
            influence = np.exp(-dist_sq / (2 * jaw_influence ** 2))  # Gaussian influence
            map_x -= dx * influence             # Apply horizontal displacement
            map_y -= dy * influence             # Apply vertical displacement

        # Apply chin extension: pixels near chin move downward with Gaussian influence
        dx_chin, dy_chin = chin_new - chin_pt
        dist_sq_chin = (map_x - chin_pt[0]) ** 2 + (map_y - chin_pt[1]) ** 2
        influence_chin = np.exp(-dist_sq_chin / (2 * chin_influence ** 2))
        map_x -= dx_chin * influence_chin
        map_y -= dy_chin * influence_chin

        # Extract ROI from frame and apply remapping to warp jaw and chin
        roi = frame[y:y+bh, x:x+bw]
        if roi.size > 0:
            frame[y:y+bh, x:x+bw] = cv.remap(
                roi, map_x - x, map_y - y,
                cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )

        return frame


    def _add_beard_effect(self, frame, jaw_pts, lower_jaw_pts, chin_pt, face_width):
        """Apply a full beard overlay on jaw, chin, and cheeks with smooth edges and gradient."""

        h, w = frame.shape[:2]  # Get frame dimensions
        
        # Compute beard parameters proportional to face size
        cheek_radius = int(face_width * self.BEARD_CHEEK_RADIUS_RATIO)      # Radius for cheek circles
        cheek_offset = int(face_width * self.BEARD_CHEEK_HEIGHT_OFFSET_RATIO)  # Vertical offset for cheeks
        chin_width = int(face_width * self.BEARD_CHIN_WIDTH_RATIO)          # Width of beard under chin
        chin_length = int(face_width * self.BEARD_CHIN_LENGTH_RATIO)        # Length of beard below chin
        blur_size = int(face_width * self.BEARD_BLUR_SIZE_RATIO)            # Kernel size for smoothing
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1      # Ensure odd kernel size
        blur_sigma = face_width * self.BEARD_BLUR_SIGMA_RATIO               # Gaussian blur sigma
        gradient_start = int(face_width * self.BEARD_GRADIENT_START_RATIO)  # Where vertical gradient starts
        gradient_range = int(face_width * self.BEARD_GRADIENT_RANGE_RATIO)  # Range over which gradient fades

        # Initialize mask (single channel) for beard
        mask = np.zeros((h, w), dtype=np.float32)

        # Fill jaw contour
        cv.fillConvexPoly(mask, cv.convexHull(jaw_pts), 1.0)

        # Extend beard onto cheeks using circles placed on jaw and lower jaw landmarks
        for pt in jaw_pts[::2]:
            cheek_pt = (pt[0], max(0, pt[1] - cheek_offset))  # Shift upward for cheek region
            cv.circle(mask, tuple(cheek_pt), cheek_radius, 1.0, -1)

        for pt in lower_jaw_pts[::2]:
            cv.circle(mask, tuple(pt), cheek_radius, 1.0, -1)

        # Extend beard below chin as a polygon
        chin_ext = np.array([
            [chin_pt[0] - chin_width, chin_pt[1]],
            [chin_pt[0] + chin_width, chin_pt[1]],
            [chin_pt[0] + chin_width + 10, chin_pt[1] + chin_length],
            [chin_pt[0] - chin_width - 10, chin_pt[1] + chin_length]
        ], dtype=np.int32)
        cv.fillConvexPoly(mask, chin_ext, 1.0)

        # Smooth edges of mask for natural blending
        mask = cv.GaussianBlur(mask, (blur_size, blur_size), blur_sigma)

        # Apply vertical gradient to fade beard upwards toward cheeks
        y_coords = np.arange(h).reshape(-1, 1)
        gradient = np.clip((y_coords - chin_pt[1] + gradient_start) / gradient_range, 0, 1)
        mask *= gradient

        # Create darkened and tinted beard overlay
        beard = (frame * self.BEARD_DARKNESS).astype(np.uint8)             # Darken underlying pixels
        tint = np.full_like(frame, self.BEARD_TINT_BGR)                    # Add subtle color tint
        beard = cv.addWeighted(beard, 0.8, tint, 0.2, 0)                  # Blend darkened frame and tint

        # Combine beard with original frame using mask
        mask_3ch = np.stack([mask] * 3, axis=-1)                           # Convert mask to 3 channels
        return ((frame * (1 - mask_3ch) + beard * mask_3ch)).astype(np.uint8)  # Blend overlay


    def _draw_debug_info(self, frame, jaw_pts, lower_jaw_pts, chin_pt):
        """Draw debug visualization for jaw and chin transformations."""
        
        # Calculate the center of the lower jaw for reference
        center = np.mean(lower_jaw_pts, axis=0).astype(np.int32)
        
        # Simulate the target jaw transformation (widening) for visualization
        new_jaw = ((lower_jaw_pts - center) * [1.4, 1.0] + center).astype(np.int32)
        
        # Simulate chin extension for visualization
        chin_new = chin_pt + [0, 8]

        # Draw original lower jaw points in blue
        for pt in lower_jaw_pts:
            cv.circle(frame, tuple(pt), 3, (255, 0, 0), -1)

        # Draw transformed jaw points in red
        for pt in new_jaw:
            cv.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
        
        # Draw original chin point in green
        cv.circle(frame, tuple(chin_pt), 5, (0, 255, 0), -1)
        
        # Draw transformed chin point in yellow
        cv.circle(frame, tuple(chin_new), 5, (0, 255, 255), -1)

        # Draw polylines connecting original and transformed jaw points for visual reference
        cv.polylines(frame, [cv.convexHull(lower_jaw_pts)], True, (255, 0, 0), 2)  # original jaw
        cv.polylines(frame, [cv.convexHull(new_jaw)], True, (0, 0, 255), 2)        # target jaw

        return frame


    def display_label(self, frame):
        """Display current mode label on frame."""
        
        # Determine label text based on mode
        mode = "DEBUG MODE" if os.environ.get("DEBUG", "0") == "1" else "WARP MODE"
        cv.putText(frame, f"Chad Jaw Filter - {mode}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        
        # In debug mode, add a legend explaining color coding
        if os.environ.get("DEBUG", "0") == "1":
            cv.putText(frame, "Blue: Original | Red: Target | Yellow: Vectors", (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        return frame

