"""
Unified Vision Neural Network (UnifiedVisionNet).
A single clean model trained from scratch for UI Display Understanding:
1. Element Type Identification (TEXT_LINE, ICON_OR_SYMBOL, CONTAINER_CARD, MEDIA_OR_IMAGE)
2. End-to-End Sequence Text Recognition (BiGRU + CTC, no character slicing)
3. UI Layout Role Classification (16 canonical roles)
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Printable ASCII characters (index 1..95, index 0 is CTC Blank)
ASCII_CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(ASCII_CHARS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(ASCII_CHARS)}
NUM_CTC_CLASSES = len(ASCII_CHARS) + 1  # 96 + 1 = 97

ELEMENT_TYPES = ["TEXT_LINE", "ICON_OR_SYMBOL", "CONTAINER_CARD", "MEDIA_OR_IMAGE"]
TYPE_TO_IDX = {t: i for i, t in enumerate(ELEMENT_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(ELEMENT_TYPES)}
NUM_ELEMENT_TYPES = len(ELEMENT_TYPES)

CANONICAL_ROLES = [
    "CONTENT_TEXT",
    "ACTION_BUTTON",
    "NAVIGATION_LINK",
    "SEARCH_BAR",
    "PROMPT_INPUT",
    "INPUT_FIELD",
    "HEADER_BAR",
    "SIDEBAR_ITEM",
    "CONTENT_CARD",
    "ICON_CONTROL",
    "STATUS_INDICATOR",
    "AVATAR_IMAGE",
    "TAB_BUTTON",
    "BANNER_HERO",
    "MENU_ITEM",
    "GENERIC_CONTAINER"
]
ROLE_TO_IDX = {r: i for i, r in enumerate(CANONICAL_ROLES)}
IDX_TO_ROLE = {i: r for i, r in enumerate(CANONICAL_ROLES)}
NUM_ROLES = len(CANONICAL_ROLES)

# Dynamic Screen Moment States
CANONICAL_MOMENTS = ["IDLE", "HOVER", "ACTIVE", "TYPING", "MUTATING", "SCROLLING"]
MOMENT_TO_IDX = {m: i for i, m in enumerate(CANONICAL_MOMENTS)}
IDX_TO_MOMENT = {i: m for i, m in enumerate(CANONICAL_MOMENTS)}
NUM_MOMENTS = len(CANONICAL_MOMENTS)

# Visual Change / State Mutation Categories
CHANGE_TYPES = ["NO_CHANGE", "CONTENT_EDIT", "STYLE_HOVER", "STATE_TOGGLE", "DISPLACEMENT_MOVE"]
CHANGE_TO_IDX = {c: i for i, c in enumerate(CHANGE_TYPES)}
IDX_TO_CHANGE = {i: c for i, c in enumerate(CHANGE_TYPES)}
NUM_CHANGES = len(CHANGE_TYPES)

# Vector Icon & UI Glyph Archetypes
ICON_ARCHETYPES = [
    "close", "minimize", "maximize", "settings", "search", "menu",
    "back", "forward", "home", "refresh", "add", "delete", "edit",
    "share", "download", "upload", "play", "pause", "stop", "mute", "more"
]
ARCHETYPE_TO_IDX = {a: i for i, a in enumerate(ICON_ARCHETYPES)}
IDX_TO_ARCHETYPE = {i: a for i, a in enumerate(ICON_ARCHETYPES)}
NUM_ARCHETYPES = len(ICON_ARCHETYPES)


class UnifiedVisionNet(nn.Module):
    """
    Unified multi-task model trained purely from display screen pixels.
    Extracts features and predicts element type, exact text sequence, and semantic role.
    """

    def __init__(self, d_feat: int = 128, rnn_hidden: int = 128):
        super().__init__()
        self.d_feat = d_feat
        self.rnn_hidden = rnn_hidden

        # ── 1. Shared Convolutional Backbone ─────────────────────────
        # Ingests [B, 1, 32, W] -> outputs [B, d_feat, 1, W // 4]
        self.conv_backbone = nn.Sequential(
            # Stage 1: 32 -> 16, W -> W/2
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Stage 2: 16 -> 8, W/2 -> W/4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Stage 3: 8 -> 2, W/4
            nn.Conv2d(64, d_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_feat),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            # Stage 4: 2 -> 1, W/4
            nn.Conv2d(d_feat, d_feat, kernel_size=(2, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_feat),
            nn.GELU()
        )

        # ── 2. Head 1: Element Type Classifier (Text vs Icon vs Card vs Media) ─
        self.type_head = nn.Sequential(
            nn.Linear(d_feat, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_ELEMENT_TYPES)
        )

        # ── 3. Head 2: Text Recognition Sequence Decoder (BiGRU + CTC) ─
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.text_head = nn.Linear(rnn_hidden * 2, NUM_CTC_CLASSES)

        # ── 4. Head 3: Layout Role Classifier (16 UI Roles) ───────────
        self.role_head = nn.Sequential(
            nn.Linear(d_feat, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_ROLES)
        )

        # ── 5. Head 4: Dynamic Screen Moment Head (IDLE, HOVER, ACTIVE, TYPING, MUTATING, SCROLLING) ─
        self.moment_head = nn.Sequential(
            nn.Linear(d_feat, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_MOMENTS)
        )

        # ── 6. Head 5: Visual Change / State Mutation Head ────────────
        # Ingests [f1, f2, |f2 - f1|] across temporal pairs -> 5 change categories
        self.change_head = nn.Sequential(
            nn.Linear(d_feat * 3, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_CHANGES)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
        return_temporal: bool = False
    ) -> Any:
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, 1, 32, W]
            x_prev: Optional prior frame tensor of shape [B, 1, 32, W] for change detection
            return_temporal: When True, returns (type_logits, ctc_logits, role_logits, moment_logits, change_logits)
        Returns:
            If return_temporal is False (default):
                (type_logits, ctc_logits, role_logits)
            If return_temporal is True:
                (type_logits, ctc_logits, role_logits, moment_logits, change_logits)
        """
        # x: [B, 1, 32, W]
        feat = self.conv_backbone(x)  # [B, d_feat, 1, W // 4]

        # 1. Global spatial pooling for patch-level classifications
        # Pool across width timesteps: [B, d_feat, 1, W//4] -> [B, d_feat]
        pooled_feat = torch.mean(feat.squeeze(2), dim=2)  # [B, d_feat]

        type_logits = self.type_head(pooled_feat)  # [B, NUM_ELEMENT_TYPES]
        role_logits = self.role_head(pooled_feat)  # [B, NUM_ROLES]

        # 2. Sequence projection along time steps for CTC text decoding
        seq_feat = feat.squeeze(2).permute(0, 2, 1)  # [B, W // 4, d_feat]
        rnn_out, _ = self.rnn(seq_feat)               # [B, W // 4, rnn_hidden * 2]
        ctc_logits = self.text_head(rnn_out)         # [B, W // 4, NUM_CTC_CLASSES]

        if not return_temporal and x_prev is None:
            return type_logits, ctc_logits, role_logits

        # 3. Dynamic Screen Moment prediction
        moment_logits = self.moment_head(pooled_feat)  # [B, NUM_MOMENTS]

        # 4. Visual State Change prediction (if prior tensor supplied)
        change_logits = None
        if x_prev is not None:
            feat_prev = self.conv_backbone(x_prev)
            pooled_prev = torch.mean(feat_prev.squeeze(2), dim=2)
            diff = torch.abs(pooled_feat - pooled_prev)
            change_in = torch.cat([pooled_prev, pooled_feat, diff], dim=-1)  # [B, d_feat * 3]
            change_logits = self.change_head(change_in)

        return type_logits, ctc_logits, role_logits, moment_logits, change_logits

    @staticmethod
    def preprocess_crop(crop: Image.Image, target_h: int = 32, min_w: int = 32, max_w: int = 384) -> torch.Tensor:
        """
        Preprocesses an arbitrary PIL image crop into a standardized float tensor [1, 1, 32, W].
        Maintains proportional aspect ratio and normalizes luminance polarity.
        """
        gray = crop.convert("L")
        w, h = gray.size
        aspect = float(w) / max(1.0, float(h))

        calc_w = int(round(target_h * aspect))
        # Snap width to multiple of 4 (conv stride compatibility)
        calc_w = ((calc_w + 3) // 4) * 4
        new_w = max(min_w, min(max_w, calc_w))

        resized = gray.resize((new_w, target_h), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0

        # Polarity normalization: text/strokes should be bright, background dark
        # If image border / background is light, invert
        border_mean = (np.mean(arr[0, :]) + np.mean(arr[-1, :]) + np.mean(arr[:, 0]) + np.mean(arr[:, -1])) / 4.0
        if border_mean > 0.45:
            arr = 1.0 - arr

        # High-contrast normalization
        arr_min, arr_max = float(np.min(arr)), float(np.max(arr))
        if arr_max - arr_min > 0.05:
            arr = (arr - arr_min) / (arr_max - arr_min)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 32, new_w]
        return tensor

    @staticmethod
    def ctc_decode(ctc_logits: torch.Tensor) -> Tuple[str, float]:
        """
        Greedy CTC decoder over [T, NUM_CTC_CLASSES] or [1, T, NUM_CTC_CLASSES].
        Returns: (decoded_string, mean_confidence)
        """
        if ctc_logits.dim() == 3:
            ctc_logits = ctc_logits.squeeze(0)

        probs = F.softmax(ctc_logits, dim=-1)
        best_indices = torch.argmax(probs, dim=-1).cpu().numpy()
        best_probs = torch.max(probs, dim=-1)[0].cpu().numpy()

        chars = []
        confidences = []
        prev_idx = 0

        for idx, conf in zip(best_indices, best_probs):
            if idx != 0 and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
                    confidences.append(float(conf))
            prev_idx = idx

        decoded = "".join(chars).strip()
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return decoded, round(mean_conf, 3)

    def predict_crop(self, crop: Image.Image, device: Optional[torch.device] = None, max_w: int = 768) -> Dict[str, Any]:
        """
        Runs complete visual inference on a single UI crop.
        Returns element type, decoded text (if text/icon), and UI semantic role.
        """
        if device is None:
            device = next(self.parameters()).device

        t = self.preprocess_crop(crop, max_w=max_w).to(device)
        self.eval()
        with torch.no_grad():
            type_logits, ctc_logits, role_logits = self(t)

            type_probs = F.softmax(type_logits, dim=-1).squeeze(0)
            role_probs = F.softmax(role_logits, dim=-1).squeeze(0)

            pred_type_idx = int(torch.argmax(type_probs).item())
            pred_type = IDX_TO_TYPE.get(pred_type_idx, "TEXT_LINE")
            type_conf = float(type_probs[pred_type_idx].item())

            pred_role_idx = int(torch.argmax(role_probs).item())
            pred_role = IDX_TO_ROLE.get(pred_role_idx, "CONTENT_TEXT")
            role_conf = float(role_probs[pred_role_idx].item())

            # Text Likelihood Gating (DEF-04)
            p_text = float(type_probs[TYPE_TO_IDX.get("TEXT_LINE", 0)].item()) if "TEXT_LINE" in TYPE_TO_IDX else 1.0
            decoded_text = ""
            text_conf = 0.0
            icon_archetype = None

            if p_text >= 0.65 and pred_type == "TEXT_LINE":
                d_text, t_conf = self.ctc_decode(ctc_logits)
                # Confidence entropy thresholding: filter single character CTC noise
                if len(d_text) > 1 or t_conf >= 0.85:
                    decoded_text = d_text
                    text_conf = t_conf
            else:
                if "CLOSE" in pred_role:
                    icon_archetype = "close"
                elif "SEARCH" in pred_role:
                    icon_archetype = "search"
                elif "MENU" in pred_role or "NAV" in pred_role:
                    icon_archetype = "menu"
                elif "TAB" in pred_role:
                    icon_archetype = "tab"
                elif "STATUS" in pred_role:
                    icon_archetype = "status_indicator"
                else:
                    icon_archetype = "icon_button"

        return {
            "predicted_type": pred_type,
            "type_confidence": round(type_conf, 3),
            "text": decoded_text,
            "text_confidence": text_conf,
            "icon_archetype": icon_archetype,
            "role": pred_role,
            "role_confidence": round(role_conf, 3)
        }

    @staticmethod
    def slice_multiline_block(img: Image.Image, min_gap: int = 3, min_line_h: int = 10) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Slices a multi-line paragraph or code block into individual horizontal line crops."""
        w, h = img.size
        if h <= 36:
            return [(img, (0, 0, w, h))]

        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        border_val = (np.mean(arr[0, :]) + np.mean(arr[-1, :]) + np.mean(arr[:, 0]) + np.mean(arr[:, -1])) / 4.0
        if border_val > 128.0:
            arr = 255.0 - arr

        row_sums = np.sum(arr, axis=1) / float(w)
        min_val = float(np.min(row_sums))
        max_val = float(np.max(row_sums))
        if max_val - min_val < 1.0:
            return [(img, (0, 0, w, h))]

        threshold = min_val + max(0.5, 0.03 * (max_val - min_val))
        is_text_row = row_sums > threshold

        lines = []
        in_line = False
        start_y = 0
        for y in range(h):
            if is_text_row[y] and not in_line:
                in_line = True
                start_y = y
            elif not is_text_row[y] and in_line:
                line_h = y - start_y
                if line_h >= min_line_h:
                    y1 = max(0, start_y - 2)
                    y2 = min(h, y + 2)
                    lines.append((img.crop((0, y1, w, y2)), (0, y1, w, y2)))
                in_line = False

        if in_line and (h - start_y) >= min_line_h:
            y1 = max(0, start_y - 2)
            lines.append((img.crop((0, y1, w, h)), (0, y1, w, h)))

        return lines if lines else [(img, (0, 0, w, h))]

    def predict_element(self, crop: Image.Image, device: Optional[torch.device] = None, max_w: int = 768) -> Dict[str, Any]:
        """
        Runs visual inference on an arbitrary UI element (word, code line, or multi-line paragraph/doc).
        Automatically handles vertical projection line-slicing for multi-line paragraphs.
        """
        w, h = crop.size
        if h > 36:
            try:
                line_slices = self.slice_multiline_block(crop)
                if len(line_slices) > 1:
                    target_dev = device if device is not None else next(self.parameters()).device
                    tensors = [self.preprocess_crop(c, target_h=32, min_w=32, max_w=max_w) for c, _ in line_slices]
                    max_batch_w = max(t.shape[3] for t in tensors)
                    padded = []
                    for t in tensors:
                        cw = t.shape[3]
                        if cw < max_batch_w:
                            pad = torch.zeros(1, 1, 32, max_batch_w, dtype=torch.float32)
                            pad[:, :, :, :cw] = t
                            padded.append(pad)
                        else:
                            padded.append(t)
                    batch_in = torch.cat(padded, dim=0).to(target_dev)
                    with torch.no_grad():
                        _, ctc_logits, _ = self(batch_in)
                    decoded = [self.ctc_decode(ctc_logits[i])[0] for i in range(len(line_slices))]
                    first_res = self.predict_crop(crop, device=target_dev, max_w=max_w)
                    return {
                        "predicted_type": "TEXT_LINE",
                        "type_confidence": 0.95,
                        "text": "\n".join(decoded),
                        "text_confidence": first_res.get("text_confidence", 0.9),
                        "role": "CONTENT_TEXT",
                        "role_confidence": 0.95
                    }
            except Exception:
                pass

        return self.predict_crop(crop, device=device, max_w=max_w)

    def predict_temporal(
        self,
        crop: Image.Image,
        crop_prev: Optional[Image.Image] = None,
        device: Optional[torch.device] = None,
        max_w: int = 384
    ) -> Dict[str, Any]:
        """
        Runs multi-task spatial and temporal inference on an element crop:
        Predicts Element Type, CTC Text, Role, Dynamic Moment State, and Visual Change Type.
        """
        if device is None:
            device = next(self.parameters()).device

        t_curr = self.preprocess_crop(crop, max_w=max_w).to(device)
        t_prev = self.preprocess_crop(crop_prev, max_w=max_w).to(device) if crop_prev is not None else None

        self.eval()
        with torch.no_grad():
            res = self(t_curr, x_prev=t_prev, return_temporal=True)
            type_logits, ctc_logits, role_logits, moment_logits, change_logits = res

            type_probs = F.softmax(type_logits, dim=-1)[0]
            pred_type_idx = int(torch.argmax(type_probs).item())
            pred_type = IDX_TO_TYPE.get(pred_type_idx, "CONTAINER_CARD")

            role_probs = F.softmax(role_logits, dim=-1)[0]
            pred_role_idx = int(torch.argmax(role_probs).item())
            pred_role = IDX_TO_ROLE.get(pred_role_idx, "CONTENT_TEXT")

            decoded_text, text_conf = self.ctc_decode(ctc_logits)

            moment_probs = F.softmax(moment_logits, dim=-1)[0]
            pred_moment_idx = int(torch.argmax(moment_probs).item())
            pred_moment = IDX_TO_MOMENT.get(pred_moment_idx, "IDLE")
            moment_conf = float(moment_probs[pred_moment_idx].item())

            change_type = "NO_CHANGE"
            change_conf = 1.0
            if change_logits is not None:
                change_probs = F.softmax(change_logits, dim=-1)[0]
                pred_change_idx = int(torch.argmax(change_probs).item())
                change_type = IDX_TO_CHANGE.get(pred_change_idx, "NO_CHANGE")
                change_conf = float(change_probs[pred_change_idx].item())

        return {
            "predicted_type": pred_type,
            "type_confidence": round(float(type_probs[pred_type_idx].item()), 3),
            "text": decoded_text,
            "text_confidence": text_conf,
            "role": pred_role,
            "role_confidence": round(float(role_probs[pred_role_idx].item()), 3),
            "moment": pred_moment,
            "moment_confidence": round(moment_conf, 3),
            "change_type": change_type,
            "change_confidence": round(change_conf, 3)
        }

    def save_weights(self, path: str):
        """Saves model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: Optional[torch.device] = None, map_location: Optional[torch.device] = None, strict: bool = False):
        """Loads model weights from disk."""
        target_dev = map_location if map_location is not None else device
        if target_dev is None:
            target_dev = next(self.parameters()).device
        state_dict = torch.load(path, map_location=target_dev, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)

