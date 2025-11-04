import numpy as np
import cv2 as cv


def overlay_png(frame, overlay_rgba, x, y, w=None, h=None):
    """
    Overlay a PNG image with alpha channel onto frame.

    Args:
        frame: Background image (BGR format)
        overlay_rgba: Overlay image with alpha channel (RGBA or RGB format)
        x: X coordinate where to place the overlay (top-left corner)
        y: Y coordinate where to place the overlay (top-left corner)
        w: Target width (optional, if None uses overlay's original width)
        h: Target height (optional, if None uses overlay's original height)

    Returns:
        Frame with overlay blended in
    """
    if overlay_rgba is None:
        return frame

    # Resize overlay if target dimensions are specified
    if w is not None and h is not None:
        overlay_rgba = cv.resize(overlay_rgba, (w, h), interpolation=cv.INTER_AREA)

    overlay_h, overlay_w = overlay_rgba.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    # Compute valid ROI in frame (handle out-of-bounds)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + overlay_w), min(frame_h, y + overlay_h)

    # If completely out of bounds, skip
    if x0 >= x1 or y0 >= y1:
        return frame

    # Corresponding crop in overlay
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    overlay_crop = overlay_rgba[oy0:oy1, ox0:ox1]

    # Extract alpha channel if it exists (RGBA), otherwise assume fully opaque
    if overlay_crop.shape[2] == 4:
        overlay_bgr = overlay_crop[..., :3]
        alpha = overlay_crop[..., 3:].astype(float) / 255.0
    else:
        overlay_bgr = overlay_crop
        alpha = np.ones((*overlay_bgr.shape[:2], 1), dtype=float)

    # Get ROI from frame
    roi = frame[y0:y1, x0:x1].astype(float)
    overlay_bgr = overlay_bgr.astype(float)

    # Alpha blend: result = alpha * overlay + (1 - alpha) * background
    blended = alpha * overlay_bgr + (1 - alpha) * roi
    frame[y0:y1, x0:x1] = blended.astype(frame.dtype)

    return frame
