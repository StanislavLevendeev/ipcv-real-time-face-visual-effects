import os
import cv2 as cv
import pygame
from utils.motion_tracking import track_gesture, is_wanted_gesture
from tasks.face_effects import FaceEffects
from tasks.face_warping import FaceWarping

class CombinedTask:
    """
    Routes frames to FaceEffects when "Pointing_Up" is detected (nerd effect)
    and to FaceWarping when "ILoveYou" is detected (chad jaw).
    Music transitions are handled efficiently.
    """

    def __init__(self):
        # Instantiate heavy submodules once
        self.face_effects = FaceEffects()
        self.face_warping = FaceWarping()

        self.current_mode = "none"
        self._current_music = None
        self._mixer_initialized = False

        # Prepare music paths once
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.music_paths = {
            "chad": os.path.join(base_dir, "music", "chad_music.wav"),
            "nerd": os.path.join(base_dir, "music", "nerd_music.mp3"),
        }

        # Initialize pygame mixer once (safe mode)
        try:
            pygame.mixer.init()
            self._mixer_initialized = True
        except Exception:
            self._mixer_initialized = False

        # Preload music files if available (prevents disk I/O later)
        if self._mixer_initialized:
            for key, path in self.music_paths.items():
                if os.path.exists(path):
                    try:
                        pygame.mixer.Sound(path)  # warm load into cache
                    except Exception:
                        pass

    def _play_music(self, mode):
        if not self._mixer_initialized:
            return

        path = self.music_paths.get(mode)
        if not path or not os.path.exists(path):
            return

        # Skip reloading if same track already playing
        if self._current_music == path and pygame.mixer.music.get_busy():
            return

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(path)
            loops = -1 if mode == "chad" else 0
            pygame.mixer.music.play(loops=loops)
            self._current_music = path
        except Exception:
            pass

    def _stop_music(self):
        if self._mixer_initialized:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._current_music = None

    def stop(self):
        """Stop everything cleanly."""
        self._stop_music()
        try:
            self.face_warping.stop()
        except Exception:
            pass
        self.current_mode = "none"

    def process_frame(self, frame):
        """Detect gestures and route frames efficiently."""
        frame_tracked, gesture_result = track_gesture(frame)

        # Combine gesture detection (single pass)
        has_love = has_point_up = False
        if gesture_result is not None:
            has_love = is_wanted_gesture(gesture_result, "ILoveYou")
            has_point_up = is_wanted_gesture(gesture_result, "Pointing_Up")

        new_mode = (
            "chad" if has_love
            else "nerd" if has_point_up
            else "none"
        )

        # Skip redundant processing if mode unchanged
        if new_mode == self.current_mode or new_mode == "none":
            if self.current_mode == "chad":
                try:
                    return self.face_warping.process_frame(frame_tracked)
                except Exception:
                    return frame_tracked
            elif self.current_mode == "nerd":
                try:
                    return self.face_effects.process_frame(frame_tracked)
                except Exception:
                    return frame_tracked
            return frame_tracked

        # Mode transition logic
        self._stop_music()
        self.current_mode = new_mode

        if new_mode == "chad":
            self._play_music("chad")
            try:
                return self.face_warping.process_frame(frame_tracked)
            except Exception:
                return frame_tracked

        elif new_mode == "nerd":
            self._play_music("nerd")
            try:
                return self.face_effects.process_frame(frame_tracked)
            except Exception:
                return frame_tracked

        # Fallback for "none"
        try:
            self.face_warping.stop()
        except Exception:
            pass
        return frame_tracked

    def display_label(self, frame):
        label_map = {
            "chad": "Chad Jaw",
            "nerd": "Nerd Effects",
            "none": "None"
        }
        cv.putText(
            frame,
            f"{label_map[self.current_mode]}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        return frame
