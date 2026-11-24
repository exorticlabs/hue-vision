"""
Hue Vision - Fast Computer Vision Element Detector
──────────────────────────────────────────────────
High-speed morphological and spatial visual element detector for DOM-absent environments
(games, remote desktops, video players, custom canvas apps, or locked displays).
Completes visual bounding-box extraction in ~10-25ms with zero external OCR dependencies.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple


class CVDetector:
    """Instant visual candidate box detector based on spatial edge and luminance morphology."""

    @staticmethod
    def detect_visual_candidates(
        image: Image.Image,
        window_rect: List[int],
        min_size: int = 12,
        max_boxes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Detects actionable visual buttons, inputs, and icon regions directly from pixels.
        Runs purely in NumPy and PIL without OpenCV dependency.
        """
        wx1, wy1, wx2, wy2 = window_rect
        img_w, img_h = image.size

        # Crop to active window region if window is smaller than full screen
        cx1 = max(0, min(wx1, img_w - 1))
        cy1 = max(0, min(wy1, img_h - 1))
        cx2 = max(cx1 + 10, min(wx2, img_w))
        cy2 = max(cy1 + 10, min(wy2, img_h))

        sub_img = image.crop((cx1, cy1, cx2, cy2))
        sub_w, sub_h = sub_img.size

        # Downscale for ultra-fast spatial gradient analysis
        scale = 0.5
        small_w = max(16, int(sub_w * scale))
        small_h = max(16, int(sub_h * scale))
        small_img = sub_img.resize((small_w, small_h), Image.Resampling.BILINEAR)

        # Convert to grayscale and compute edge gradient
        gray = small_img.convert("L")
        arr = np.array(gray, dtype=np.float32)

        # Sobel-like gradient approximation using horizontal and vertical diffs
        dx = np.abs(arr[:, 1:] - arr[:, :-1])
        dy = np.abs(arr[1:, :] - arr[:-1, :])
        grad = dx[:-1, :] + dy[:, :-1]

        # Threshold gradient into binary edges
        edge_threshold = max(15.0, float(np.mean(grad) * 1.5))
        edges = (grad > edge_threshold).astype(np.uint8)

        # Find bounding boxes using horizontal projection profiles
        candidates = []
        h_proj = np.sum(edges, axis=1)
        row_indices = np.where(h_proj > (small_w * 0.02))[0]

        if len(row_indices) > 0:
            diffs = np.diff(row_indices)
            split_points = np.where(diffs > 2)[0]
            bands = []
            start = row_indices[0]
            for sp in split_points:
                end = row_indices[sp]
                if end - start >= int(min_size * scale * 0.5):
                    bands.append((start, end))
                start = row_indices[sp + 1]
            if row_indices[-1] - start >= int(min_size * scale * 0.5):
                bands.append((start, row_indices[-1]))

            for b_top, b_bot in bands[:15]:
                band_edges = edges[b_top:b_bot, :]
                v_proj = np.sum(band_edges, axis=0)
                col_indices = np.where(v_proj > 0)[0]
                if len(col_indices) == 0:
                    continue

                col_diffs = np.diff(col_indices)
                col_splits = np.where(col_diffs > 4)[0]
                c_start = col_indices[0]
                for csp in col_splits:
                    c_end = col_indices[csp]
                    box_w = int((c_end - c_start) / scale)
                    box_h = int((b_bot - b_top) / scale)
                    if box_w >= min_size and box_h >= min_size:
                        abs_x1 = int(cx1 + (c_start / scale))
                        abs_y1 = int(cy1 + (b_top / scale))
                        candidates.append({
                            "bbox": [abs_x1, abs_y1, abs_x1 + box_w, abs_y1 + box_h],
                            "control_type": "CustomControl",
                            "is_interactive": True
                        })
                    c_start = col_indices[csp + 1]

                box_w = int((col_indices[-1] - c_start) / scale)
                box_h = int((b_bot - b_top) / scale)
                if box_w >= min_size and box_h >= min_size:
                    abs_x1 = int(cx1 + (c_start / scale))
                    abs_y1 = int(cy1 + (b_top / scale))
                    candidates.append({
                        "bbox": [abs_x1, abs_y1, abs_x1 + box_w, abs_y1 + box_h],
                        "control_type": "CustomControl",
                        "is_interactive": True
                    })

                if len(candidates) >= max_boxes:
                    break

        return candidates[:max_boxes]
