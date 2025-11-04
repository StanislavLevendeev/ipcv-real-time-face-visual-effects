# ...existing code...
import numpy as np
import cv2 as cv

overlay_cache = {
    "glasses": {"size": None, "rgb": None, "alpha": None},
    "teeth": {"size": None, "rgb": None, "alpha": None},
    "plaster": {"size": None, "rgb": None, "alpha": None},
}


def overlay_png(frame, overlay_rgba, x, y, w=None, h=None, cache_key=None):
    """
    Overlay a PNG image with alpha channel onto frame.

    Args:
        frame: Background image (BGR format)
        overlay_rgba: Overlay image with alpha channel (RGBA or RGB format)
        x: X coordinate where to place the overlay (top-left corner)
        y: Y coordinate where to place the overlay (top-left corner)
        w: Target width (optional, if None uses overlay's original width)
        h: Target height (optional, if None uses overlay's original height)
        cache_key: optional string key to cache the resized overlay (rgb + alpha)
                   for repeated reuse (improves performance)

    Returns:
        Frame with overlay blended in
    """
    if overlay_rgba is None:
        return frame

    base_h, base_w = overlay_rgba.shape[:2]
    # determine target size
    if w is None:
        target_w = base_w
    else:
        target_w = int(w)
    if h is None:
        target_h = base_h
    else:
        target_h = int(h)

    if target_w <= 0 or target_h <= 0:
        return frame

    cached_rgb = cached_alpha = None
    if cache_key is not None:
        cache_entry = overlay_cache.get(cache_key)
        if cache_entry:
            if cache_entry["size"] == (target_w, target_h):
                cached_rgb = cache_entry["rgb"]
                cached_alpha = cache_entry["alpha"]

    if cached_rgb is None or cached_alpha is None:
        # choose interpolation depending on scaling
        interp = (
            cv.INTER_AREA
            if (target_w < base_w or target_h < base_h)
            else cv.INTER_LINEAR
        )
        overlay_resized = cv.resize(
            overlay_rgba, (target_w, target_h), interpolation=interp
        )

        # extract alpha and rgb, convert to float32
        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3:4].astype(np.float32) / 255.0
            overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)
        else:
            alpha = np.ones((target_h, target_w, 1), dtype=np.float32)
            overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)

        if cache_key is not None:
            # ensure dictionary entry exists and store precomputed arrays
            overlay_cache.setdefault(
                cache_key, {"size": None, "rgb": None, "alpha": None}
            )
            overlay_cache[cache_key]["size"] = (target_w, target_h)
            overlay_cache[cache_key]["rgb"] = overlay_rgb
            overlay_cache[cache_key]["alpha"] = alpha
    else:
        overlay_rgb = cached_rgb
        alpha = cached_alpha

    frame_h, frame_w = frame.shape[:2]

    # Compute valid ROI in frame (handle out-of-bounds)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + target_w), min(frame_h, y + target_h)

    # If completely out of bounds, skip
    if x0 >= x1 or y0 >= y1:
        return frame

    # Corresponding crop in overlay
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    overlay_region = overlay_rgb[oy0:oy1, ox0:ox1]
    alpha_region = alpha[oy0:oy1, ox0:ox1]

    # Get ROI from frame and blend
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    blended = alpha_region * overlay_region + (1.0 - alpha_region) * roi
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    return frame
