"""Chad-jaw face-warping effect.

Runs MediaPipe FaceMesh on each frame and applies three Gaussian-weighted
landmark warps (jaw widening + chin extension, eyebrow frown, beard overlay)
plus color grading and a periodic flash. Parameters are expressed as ratios
of the detected face width so the effect scales with distance from the camera.
"""

import os

import cv2 as cv
import mediapipe as mp
import numpy as np


class FaceWarping:
    # Color grading
    SATURATION = 2.0
    BRIGHTNESS = 1.1
    CONTRAST = 1.3

    # Flash
    ENABLE_FLASH = True
    FLASH_INTERVAL = 10
    FLASH_DURATION = 3
    FLASH_INTENSITY = 1

    # Eyebrow frown (ratios of face width)
    EYEBROW_VERTICAL_SHIFT_RATIO = 0.055
    EYEBROW_HORIZONTAL_SHIFT_RATIO = 0.038
    EYEBROW_OUTER_LIFT_RATIO = 0.020
    EYEBROW_INFLUENCE_RATIO = 0.050
    EYEBROW_ROI_MARGIN_RATIO = 0.125

    # Jaw widening / chin extension
    JAW_WIDTH_SCALE = 1.35
    CHIN_EXTENSION_RATIO = 0.013
    JAW_INFLUENCE_RATIO = 0.055
    CHIN_INFLUENCE_RATIO = 0.030
    JAW_ROI_MARGIN_RATIO = 0.200

    # Beard overlay
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

    # FaceMesh landmark indices
    JAW_INDICES = [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    ]
    LOWER_JAW_INDICES = [
        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    ]
    CHIN_INDEX = 152
    LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300]

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    # --- Pipeline -------------------------------------------------------

    def process_frame(self, frame):
        h, w, _ = frame.shape

        frame = self._apply_color_effects(frame)
        # Force monochrome: grade colors first, then drop them. Grading a grey
        # frame would be a no-op, so order matters.
        frame = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

        result = self.face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return self._apply_flash_effect(frame)

        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])

        jaw_pts = points[self.JAW_INDICES]
        lower_jaw_pts = points[self.LOWER_JAW_INDICES]
        chin_pt = points[self.CHIN_INDEX]
        left_brow = points[self.LEFT_EYEBROW_INDICES]
        right_brow = points[self.RIGHT_EYEBROW_INDICES]

        face_width = np.max(jaw_pts[:, 0]) - np.min(jaw_pts[:, 0])

        if self._debug():
            frame = self._draw_debug_info(frame, jaw_pts, lower_jaw_pts, chin_pt)

        frame = self._apply_jaw_warp(frame, lower_jaw_pts, chin_pt, face_width)
        frame = self._add_beard_effect(frame, jaw_pts, lower_jaw_pts, chin_pt, face_width)
        frame = self._apply_eyebrow_frown(frame, left_brow, right_brow, face_width)

        return self._apply_flash_effect(frame)

    def stop(self):
        """Reset transient state when the task is switched away."""
        self.frame_count = 0
        self.flash_active = False
        self.flash_timer = 0

    def display_label(self, frame):
        mode = "DEBUG MODE" if self._debug() else "WARP MODE"
        cv.putText(
            frame,
            f"Chad Jaw Filter - {mode}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        if self._debug():
            cv.putText(
                frame,
                "Blue: Original | Red: Target | Yellow: Vectors",
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
        return frame

    # --- Effects --------------------------------------------------------

    def _apply_color_effects(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.SATURATION, 0, 255)
        frame = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR).astype(np.float32)
        # Standard linear contrast/brightness: out = c * (b * px - 128) + 128.
        frame = self.CONTRAST * (frame * self.BRIGHTNESS - 128) + 128
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _apply_flash_effect(self, frame):
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
        h, w = frame.shape[:2]

        v_shift = int(face_width * self.EYEBROW_VERTICAL_SHIFT_RATIO)
        h_shift = int(face_width * self.EYEBROW_HORIZONTAL_SHIFT_RATIO)
        outer_lift = int(face_width * self.EYEBROW_OUTER_LIFT_RATIO)
        influence_radius = int(face_width * self.EYEBROW_INFLUENCE_RATIO)
        roi_margin = int(face_width * self.EYEBROW_ROI_MARGIN_RATIO)

        # Target landmark positions: inner points pushed down-and-in (angry V),
        # outer points lifted to exaggerate the arch.
        new_left = left_brow.astype(np.float32).copy()
        new_right = right_brow.astype(np.float32).copy()
        n = len(left_brow)
        for i in range(n):
            factor = (n - i) / n
            new_left[i, 1] += v_shift * factor
            new_left[i, 0] += h_shift * factor
            new_right[i, 1] += v_shift * factor
            new_right[i, 0] -= h_shift * factor
            if i > 2:
                new_left[i, 1] -= outer_lift
                new_right[i, 1] -= outer_lift

        x, y, bw, bh = cv.boundingRect(
            np.vstack([left_brow, right_brow]).astype(np.int32)
        )
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)

        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x, np.arange(bh, dtype=np.float32) + y
        )

        # Gaussian-weighted displacement: pixels close to each landmark inherit
        # its movement, influence decaying with distance squared.
        radius_sq = 2 * influence_radius ** 2
        for src, dst in zip(
            np.vstack([left_brow, right_brow]),
            np.vstack([new_left, new_right]),
        ):
            dx, dy = dst - src
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2
            influence = np.exp(-dist_sq / radius_sq)
            map_x -= dx * influence
            map_y -= dy * influence

        roi = frame[y : y + bh, x : x + bw]
        if roi.size > 0:
            frame[y : y + bh, x : x + bw] = cv.remap(
                roi, map_x - x, map_y - y, cv.INTER_LINEAR,
                borderMode=cv.BORDER_REPLICATE,
            )
        return frame

    def _apply_jaw_warp(self, frame, lower_jaw_pts, chin_pt, face_width):
        h, w = frame.shape[:2]

        chin_ext = int(face_width * self.CHIN_EXTENSION_RATIO)
        jaw_influence = face_width * self.JAW_INFLUENCE_RATIO
        chin_influence = face_width * self.CHIN_INFLUENCE_RATIO
        roi_margin = int(face_width * self.JAW_ROI_MARGIN_RATIO)

        center = np.mean(lower_jaw_pts, axis=0)
        new_jaw = (
            (lower_jaw_pts - center) * [self.JAW_WIDTH_SCALE, 1.0] + center
        ).astype(np.float32)
        chin_new = chin_pt.astype(np.float32) + [0, chin_ext]

        all_pts = np.vstack([lower_jaw_pts, [chin_pt]])
        x, y, bw, bh = cv.boundingRect(all_pts.astype(np.int32))
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        bw = min(w - x, bw + 2 * roi_margin)
        bh = min(h - y, bh + 2 * roi_margin)

        map_x, map_y = np.meshgrid(
            np.arange(bw, dtype=np.float32) + x, np.arange(bh, dtype=np.float32) + y
        )

        for src, dst in zip(lower_jaw_pts, new_jaw):
            dx, dy = dst - src
            dist_sq = (map_x - src[0]) ** 2 + (map_y - src[1]) ** 2
            influence = np.exp(-dist_sq / (2 * jaw_influence ** 2))
            map_x -= dx * influence
            map_y -= dy * influence

        dx_chin, dy_chin = chin_new - chin_pt
        dist_sq_chin = (map_x - chin_pt[0]) ** 2 + (map_y - chin_pt[1]) ** 2
        influence_chin = np.exp(-dist_sq_chin / (2 * chin_influence ** 2))
        map_x -= dx_chin * influence_chin
        map_y -= dy_chin * influence_chin

        roi = frame[y : y + bh, x : x + bw]
        if roi.size > 0:
            frame[y : y + bh, x : x + bw] = cv.remap(
                roi, map_x - x, map_y - y, cv.INTER_LINEAR,
                borderMode=cv.BORDER_REPLICATE,
            )
        return frame

    def _add_beard_effect(self, frame, jaw_pts, lower_jaw_pts, chin_pt, face_width):
        h, w = frame.shape[:2]

        cheek_radius = int(face_width * self.BEARD_CHEEK_RADIUS_RATIO)
        cheek_offset = int(face_width * self.BEARD_CHEEK_HEIGHT_OFFSET_RATIO)
        chin_width = int(face_width * self.BEARD_CHIN_WIDTH_RATIO)
        chin_length = int(face_width * self.BEARD_CHIN_LENGTH_RATIO)
        blur_size = int(face_width * self.BEARD_BLUR_SIZE_RATIO)
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        blur_sigma = face_width * self.BEARD_BLUR_SIGMA_RATIO
        gradient_start = int(face_width * self.BEARD_GRADIENT_START_RATIO)
        gradient_range = int(face_width * self.BEARD_GRADIENT_RANGE_RATIO)

        # Build a soft mask: jaw hull + cheek circles + rectangular extension
        # below the chin. Blurred and vertically graded so the top edge fades
        # into the cheeks instead of ending in a hard line.
        mask = np.zeros((h, w), dtype=np.float32)
        cv.fillConvexPoly(mask, cv.convexHull(jaw_pts), 1.0)
        for pt in jaw_pts[::2]:
            cv.circle(mask, (pt[0], max(0, pt[1] - cheek_offset)), cheek_radius, 1.0, -1)
        for pt in lower_jaw_pts[::2]:
            cv.circle(mask, tuple(pt), cheek_radius, 1.0, -1)

        chin_box = np.array(
            [
                [chin_pt[0] - chin_width, chin_pt[1]],
                [chin_pt[0] + chin_width, chin_pt[1]],
                [chin_pt[0] + chin_width + 10, chin_pt[1] + chin_length],
                [chin_pt[0] - chin_width - 10, chin_pt[1] + chin_length],
            ],
            dtype=np.int32,
        )
        cv.fillConvexPoly(mask, chin_box, 1.0)

        mask = cv.GaussianBlur(mask, (blur_size, blur_size), blur_sigma)

        y_coords = np.arange(h).reshape(-1, 1)
        gradient = np.clip(
            (y_coords - chin_pt[1] + gradient_start) / gradient_range, 0, 1
        )
        mask *= gradient

        beard = (frame * self.BEARD_DARKNESS).astype(np.uint8)
        tint = np.full_like(frame, self.BEARD_TINT_BGR)
        beard = cv.addWeighted(beard, 0.8, tint, 0.2, 0)

        mask_3ch = np.stack([mask] * 3, axis=-1)
        return (frame * (1 - mask_3ch) + beard * mask_3ch).astype(np.uint8)

    # --- Debug ----------------------------------------------------------

    def _debug(self) -> bool:
        # Env var wins when set; otherwise fall back to the constructor flag.
        env = os.getenv("DEBUG")
        if env is not None:
            return env == "1"
        return self.debug_mode

    def _draw_debug_info(self, frame, jaw_pts, lower_jaw_pts, chin_pt):
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
