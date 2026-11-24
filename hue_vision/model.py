"""
Hue Vision - Unified Vision Model Manager
─────────────────────────────────────────
Self-contained, production-grade visual inference engine powered by UnifiedVisionNet.
Supports batched GPU (CUDA/MPS) and CPU inference with automatic precision management,
with graceful CPU fallback when neural accelerators are absent.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

# Import core definitions from hue_vision.unified_vision_model
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from hue_vision.unified_vision_model import (
        UnifiedVisionNet,
        ELEMENT_TYPES,
        TYPE_TO_IDX,
        IDX_TO_TYPE,
        CANONICAL_ROLES,
        ROLE_TO_IDX,
        IDX_TO_ROLE,
        CHAR_TO_IDX,
        IDX_TO_CHAR,
        CANONICAL_MOMENTS,
        MOMENT_TO_IDX,
        IDX_TO_MOMENT,
        CHANGE_TYPES,
        CHANGE_TO_IDX,
        IDX_TO_CHANGE
    )
    MODEL_AVAILABLE = True
except ImportError:
    # Standalone fallbacks if torch or model file is unavailable
    ELEMENT_TYPES = [
        "BUTTON", "INPUT_FIELD", "TEXT_LINE", "PARAGRAPH", "ICON",
        "CONTAINER_CARD", "NAVBAR_HEADER", "MENU_ITEM", "LIST_ROW",
        "TAB_BUTTON", "BADGE_CHIP", "MODAL_POPUP", "IMAGE_THUMB", "SLIDER_SWITCH"
    ]
    TYPE_TO_IDX = {t: i for i, t in enumerate(ELEMENT_TYPES)}
    IDX_TO_TYPE = {i: t for i, t in enumerate(ELEMENT_TYPES)}

    CANONICAL_ROLES = [
        "ACTION_BUTTON", "NAV_MENU_BUTTON", "CLOSE_DISMISS_BUTTON", "TAB_BUTTON",
        "SEARCH_BAR", "PROMPT_INPUT", "FILTER_INPUT", "FORM_FIELD",
        "SECTION_HEADER", "CONTENT_BODY", "STAT_COUNTER", "CAPTION_HINT",
        "APP_ICON", "ACTION_ICON", "STATUS_INDICATOR", "USER_AVATAR",
        "CONTENT_CARD", "LIST_ITEM", "TABLE_ROW", "DROPDOWN_MENU",
        "DIALOG_MODAL", "ALERT_BANNER", "FLOATING_PANEL", "GENERIC_CONTAINER"
    ]
    ROLE_TO_IDX = {r: i for i, r in enumerate(CANONICAL_ROLES)}
    IDX_TO_ROLE = {i: r for i, r in enumerate(CANONICAL_ROLES)}

    CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-+*/=():;\"'@#$%^&_[]{}|<>"
    CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
    IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

    CANONICAL_MOMENTS = ["NORMAL", "HOVER", "ACTIVE", "FOCUS", "TYPING"]
    MOMENT_TO_IDX = {m: i for i, m in enumerate(CANONICAL_MOMENTS)}
    IDX_TO_MOMENT = {i: m for i, m in enumerate(CANONICAL_MOMENTS)}

    CHANGE_TYPES = ["NO_CHANGE", "MUTATED", "TRANSIENT", "APPEARED", "DISAPPEARED"]
    CHANGE_TO_IDX = {c: i for i, c in enumerate(CHANGE_TYPES)}
    IDX_TO_CHANGE = {i: c for i, c in enumerate(CHANGE_TYPES)}

    UnifiedVisionNet = None
    MODEL_AVAILABLE = False


class VisionModelManager:
    """Manages UnifiedVisionNet loading, caching, and batched inference."""

    _instance: Optional["VisionModelManager"] = None

    def __init__(self, weights_path: Optional[str] = None, device_str: Optional[str] = None):
        self.device = None
        self.model = None

        if TORCH_AVAILABLE and MODEL_AVAILABLE and UnifiedVisionNet is not None:
            if device_str:
                self.device = torch.device(device_str)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            try:
                self.model = UnifiedVisionNet().to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"[!] Warning: Could not initialize UnifiedVisionNet: {e}", file=sys.stderr)
                self.model = None

            self.weights_path = weights_path or self._discover_best_weights()
            if self.model is not None and self.weights_path and os.path.exists(self.weights_path):
                self._load_weights(self.weights_path)
            elif self.model is not None:
                print(f"[!] Warning: No weights file found at '{self.weights_path}'. Running with initialized weights.", file=sys.stderr)
        else:
            self.weights_path = None

    @classmethod
    def get_instance(cls, weights_path: Optional[str] = None) -> "VisionModelManager":
        """Singleton accessor for fast model reuse across tool calls."""
        if cls._instance is None:
            cls._instance = VisionModelManager(weights_path=weights_path)
        return cls._instance

    def _discover_best_weights(self) -> Optional[str]:
        """Locates the highest-priority trained checkpoint on disk."""
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(pkg_dir)

        candidates = [
            os.path.join(pkg_dir, "weights", "unified_vision_net_production.pt"),
            os.path.join(pkg_dir, "weights", "unified_vision_net.pt"),
            os.path.join(repo_root, "weights", "unified_vision_net_production.pt"),
            os.path.join(repo_root, "weights", "unified_vision_net.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_weights(self, path: str) -> None:
        try:
            self.model.load_weights(path, map_location=self.device)
            print(f"[+] Loaded Hue Vision model weights from: {path} (Device: {self.device})", file=sys.stderr)
        except Exception as e:
            print(f"[!] Failed to load model weights from {path}: {e}", file=sys.stderr)

    def predict_crops(
        self,
        crops: List[Image.Image],
        prev_crops: Optional[List[Image.Image]] = None,
        batch_size: int = 64
    ) -> List[Dict[str, Any]]:
        """
        Runs complete multi-task visual inference across a batch of UI element crops.
        Returns: List of predicted {type, role, text, text_confidence, moment, state_change}.
        """
        if not crops:
            return []

        # Graceful fallback when PyTorch or weights are not loaded
        if self.model is None or not TORCH_AVAILABLE or UnifiedVisionNet is None:
            return [
                {
                    "element_type": "BUTTON" if c.size[0] > 20 and c.size[1] > 20 else "ICON",
                    "semantic_role": "ACTION_BUTTON",
                    "ocr_text": "",
                    "text_confidence": 0.0,
                    "icon_archetype": "icon_button",
                    "moment": "NORMAL",
                    "state_change": "NO_CHANGE"
                }
                for c in crops
            ]

        results = []
        target_h = 32
        default_w = 192

        has_prev = bool(prev_crops and len(prev_crops) == len(crops))

        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            prev_chunk = prev_crops[i:i + batch_size] if has_prev else None

            # Prepare tensor batch
            tensors = []
            prev_tensors = []

            for idx, c in enumerate(chunk):
                t = UnifiedVisionNet.preprocess_crop(c, target_h=target_h, min_w=16, max_w=default_w)
                cur_w = t.shape[3]
                if cur_w < default_w:
                    padded = torch.zeros(1, 1, target_h, default_w, dtype=torch.float32)
                    padded[:, :, :, :cur_w] = t
                    t = padded
                else:
                    t = t[:, :, :, :default_w]
                tensors.append(t)

                if has_prev:
                    pc = prev_chunk[idx]
                    tp = UnifiedVisionNet.preprocess_crop(pc, target_h=target_h, min_w=16, max_w=default_w)
                    cur_wp = tp.shape[3]
                    if cur_wp < default_w:
                        padded_p = torch.zeros(1, 1, target_h, default_w, dtype=torch.float32)
                        padded_p[:, :, :, :cur_wp] = tp
                        tp = padded_p
                    else:
                        tp = tp[:, :, :, :default_w]
                    prev_tensors.append(tp)

            batch_x = torch.cat(tensors, dim=0).to(self.device)
            batch_prev = torch.cat(prev_tensors, dim=0).to(self.device) if has_prev else None

            with torch.no_grad():
                autocast_enabled = (self.device.type == "cuda")
                if autocast_enabled:
                    with torch.amp.autocast("cuda"):
                        pred_types, pred_ctc, pred_roles, pred_moments, pred_changes = self.model(
                            batch_x,
                            x_prev=batch_prev,
                            return_temporal=True
                        )
                else:
                    pred_types, pred_ctc, pred_roles, pred_moments, pred_changes = self.model(
                        batch_x,
                        x_prev=batch_prev,
                        return_temporal=True
                    )

            type_probs = F.softmax(pred_types, dim=-1)
            type_preds = pred_types.argmax(dim=-1).cpu().numpy()
            role_preds = pred_roles.argmax(dim=-1).cpu().numpy()
            moment_preds = pred_moments.argmax(dim=-1).cpu().numpy()
            change_preds = pred_changes.argmax(dim=-1).cpu().numpy() if pred_changes is not None else None

            text_type_idx = TYPE_TO_IDX.get("TEXT_LINE", 0)
            p_text_batch = type_probs[:, text_type_idx].cpu().numpy() if text_type_idx < type_probs.shape[-1] else np.ones(len(chunk))

            for b_idx in range(len(chunk)):
                etype = IDX_TO_TYPE.get(int(type_preds[b_idx]), "UNKNOWN")
                erole = IDX_TO_ROLE.get(int(role_preds[b_idx]), "GENERIC_CONTAINER")
                emoment = IDX_TO_MOMENT.get(int(moment_preds[b_idx]), "NORMAL")
                echange = IDX_TO_CHANGE.get(int(change_preds[b_idx]), "NO_CHANGE") if change_preds is not None else "NO_CHANGE"
                p_text = float(p_text_batch[b_idx])

                # Gated Text vs Icon Archetype Recognition
                decoded_text = ""
                text_conf = 0.0
                icon_archetype = None

                if p_text >= 0.65 and etype in ["TEXT_LINE", "INPUT_FIELD", "BUTTON"]:
                    ctc_sub = pred_ctc[b_idx].float()
                    d_text, t_conf = UnifiedVisionNet.ctc_decode(ctc_sub)
                    # Entropy filter: prevent CTC from emitting single-character visual noise
                    if len(d_text) > 1 or t_conf >= 0.85:
                        decoded_text = d_text
                        text_conf = t_conf
                else:
                    # Classify icon archetype
                    if "CLOSE" in erole:
                        icon_archetype = "close"
                    elif "SEARCH" in erole:
                        icon_archetype = "search"
                    elif "MENU" in erole or "NAV" in erole:
                        icon_archetype = "menu"
                    elif "TAB" in erole:
                        icon_archetype = "tab"
                    elif "STATUS" in erole:
                        icon_archetype = "status_indicator"
                    else:
                        icon_archetype = "icon_button"

                results.append({
                    "element_type": etype,
                    "semantic_role": erole,
                    "ocr_text": decoded_text,
                    "text_confidence": text_conf,
                    "icon_archetype": icon_archetype,
                    "moment": emoment,
                    "state_change": echange
                })

        return results
