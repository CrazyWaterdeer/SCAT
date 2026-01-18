"""
Deposit detection module using adaptive thresholding and optional deep learning.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class Deposit:
    """Represents a single detected deposit."""
    id: int
    contour: np.ndarray
    x: int
    y: int
    width: int
    height: int
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    centroid: Tuple[int, int]
    
    # Color features (filled after extraction)
    mean_hue: float = 0.0
    mean_saturation: float = 0.0
    mean_lightness: float = 0.0
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    iod: float = 0.0
    
    # Classification (filled after classification)
    label: str = "unknown"  # "rod", "normal", "artifact"
    confidence: float = 0.0
    
    # Labeling metadata
    merged: bool = False      # True if this was created by merging multiple deposits
    group_id: Optional[int] = None  # Group ID for logically grouped deposits
    
    def get_patch(self, image: np.ndarray, padding: int = 5) -> np.ndarray:
        """Extract image patch for this deposit."""
        h, w = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w, self.x + self.width + padding)
        y2 = min(h, self.y + self.height + padding)
        return image[y1:y2, x1:x2]


class DepositDetector:
    """Detects deposits in fly excreta images using adaptive thresholding or U-Net."""
    
    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 10000,
        block_size: int = 51,
        c_value: int = 10,
        edge_margin: int = 20,
        sensitive_mode: bool = False,
        min_circularity: float = 0.3,
        unet_model_path: Optional[str] = None
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.block_size = block_size
        self.c_value = c_value
        self.edge_margin = edge_margin
        self.sensitive_mode = sensitive_mode
        self.min_circularity = min_circularity
        
        # U-Net model (lazy loaded)
        self.unet_model_path = unet_model_path
        self._unet_detector = None
    
    def _get_unet_detector(self):
        """Lazy load U-Net detector."""
        if self._unet_detector is None and self.unet_model_path:
            from .segmentation import UNetDetector
            self._unet_detector = UNetDetector(Path(self.unet_model_path))
        return self._unet_detector
    
    def detect(self, image: np.ndarray) -> List[Deposit]:
        """Detect deposits in image."""
        # If U-Net model is available, use it
        unet = self._get_unet_detector()
        if unet is not None:
            contours = unet.detect(image, min_area=self.min_area)
            return self._process_contours(contours, image.shape[:2])
        
        # Otherwise use traditional detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if self.sensitive_mode:
            # Multi-level detection for dilute deposits
            all_contours = self._multi_level_detect(gray, image)
        else:
            # Standard single-level detection
            all_contours = self._single_level_detect(gray)
        
        deposits = self._process_contours(all_contours, image.shape[:2])
        return deposits
    
    def _single_level_detect(self, gray: np.ndarray) -> List[np.ndarray]:
        """Standard adaptive threshold detection."""
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size, 
            self.c_value
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def _multi_level_detect(self, gray: np.ndarray, image: np.ndarray) -> List[np.ndarray]:
        """
        Two-stage detection for sensitive mode.
        Stage 1: Standard detection for solid deposits (precise boundaries)
        Stage 2: Sensitive detection ONLY in areas not already detected (for dilute deposits)
        """
        h, w = gray.shape[:2]
        
        # ===== STAGE 1: Standard detection (precise boundaries for solid deposits) =====
        thresh_standard = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c_value
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_standard = cv2.morphologyEx(thresh_standard, cv2.MORPH_OPEN, kernel)
        thresh_standard = cv2.morphologyEx(thresh_standard, cv2.MORPH_CLOSE, kernel)
        
        # Find contours from standard detection
        contours_standard, _ = cv2.findContours(
            thresh_standard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create mask of already-detected regions (with some padding)
        detected_mask = np.zeros((h, w), dtype=np.uint8)
        for cnt in contours_standard:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                # Fill the detected region with padding
                cv2.drawContours(detected_mask, [cnt], -1, 255, -1)
        
        # Dilate to create exclusion zone around detected deposits
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        exclusion_mask = cv2.dilate(detected_mask, dilate_kernel, iterations=1)
        
        # ===== STAGE 2: Sensitive detection in undetected areas only =====
        sensitive_masks = []
        
        # More sensitive adaptive threshold
        c_sensitive = max(2, self.c_value - 5)
        thresh_sensitive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            c_sensitive
        )
        sensitive_masks.append(thresh_sensitive)
        
        # Color-based detection for dilute deposits
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            
            mean_v = np.mean(v_channel)
            std_v = np.std(v_channel)
            
            # Regions slightly darker than background
            threshold_v = int(mean_v - 0.4 * std_v)
            _, thresh_v = cv2.threshold(v_channel, threshold_v, 255, cv2.THRESH_BINARY_INV)
            sensitive_masks.append(thresh_v)
            
            # Colored regions (saturation)
            _, thresh_s = cv2.threshold(s_channel, 25, 255, cv2.THRESH_BINARY)
            colored_mask = cv2.bitwise_and(thresh_s, thresh_v)
            sensitive_masks.append(colored_mask)
        
        # Combine sensitive masks
        combined_sensitive = np.zeros_like(gray)
        for mask in sensitive_masks:
            combined_sensitive = cv2.bitwise_or(combined_sensitive, mask)
        
        # Light cleanup for sensitive detection
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_sensitive = cv2.morphologyEx(combined_sensitive, cv2.MORPH_OPEN, kernel_small)
        combined_sensitive = cv2.morphologyEx(combined_sensitive, cv2.MORPH_CLOSE, kernel_small)
        
        # CRITICAL: Remove already-detected regions from sensitive results
        combined_sensitive = cv2.bitwise_and(combined_sensitive, cv2.bitwise_not(exclusion_mask))
        
        # Find contours from sensitive detection (new deposits only)
        contours_sensitive, _ = cv2.findContours(
            combined_sensitive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # ===== Combine results =====
        # Standard contours first (priority), then sensitive contours
        all_contours = list(contours_standard) + list(contours_sensitive)
        
        # Remove any remaining duplicates
        all_contours = self._remove_duplicate_contours(all_contours)
        
        return all_contours
    
    def _remove_duplicate_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Remove significantly overlapping contours, keeping the larger one."""
        if len(contours) <= 1:
            return contours
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        keep = []
        kept_centers = []  # Store (cx, cy, area) for kept contours
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if this centroid is too close to any kept contour's center
            # Use distance-based check instead of polygon containment
            is_duplicate = False
            for kept_cx, kept_cy, kept_area in kept_centers:
                dist = np.sqrt((cx - kept_cx)**2 + (cy - kept_cy)**2)
                # Consider duplicate if centers are very close (within ~half the radius)
                min_radius = np.sqrt(min(area, kept_area) / np.pi)
                if dist < min_radius * 0.7:  # More tolerant threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(cnt)
                kept_centers.append((cx, cy, area))
        
        return keep
    
    def _process_contours(
        self, 
        contours: List[np.ndarray], 
        image_shape: Tuple[int, int]
    ) -> List[Deposit]:
        """Process contours into Deposit objects."""
        deposits = []
        h, w = image_shape
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            if (x < self.edge_margin or y < self.edge_margin or
                x + bw > w - self.edge_margin or y + bh > h - self.edge_margin):
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Note: Do NOT filter by circularity here - let the ML classifier handle it
            # This ensures rod-shaped deposits are not incorrectly filtered out
            
            aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
            
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2
            
            deposit = Deposit(
                id=len(deposits) + 1,  # Start from 1, not 0
                contour=cnt,
                x=x, y=y,
                width=bw, height=bh,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                centroid=(cx, cy)
            )
            deposits.append(deposit)
        
        return deposits
    
    def get_binary_mask(self, image: np.ndarray) -> np.ndarray:
        """Return binary mask of detected regions."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c_value
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
