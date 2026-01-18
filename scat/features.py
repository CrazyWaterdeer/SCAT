"""
Feature extraction module for deposit analysis.
"""

import cv2
import numpy as np
from typing import List
from .detector import Deposit


class FeatureExtractor:
    """Extracts color and morphological features from detected deposits."""
    
    def __init__(self, dpi: float = 600.0):
        self.dpi = dpi
        self.pixels_per_um = dpi / 25400
    
    def extract_features(self, image: np.ndarray, deposits: List[Deposit]) -> List[Deposit]:
        """Extract color features for all deposits.
        
        Optimized: Reuses mask array instead of reallocating for each deposit.
        """
        if not deposits:
            return deposits
        
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        
        # Pre-allocate mask once (memory optimization)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for deposit in deposits:
            # Clear and reuse mask instead of reallocating
            mask.fill(0)
            cv2.drawContours(mask, [deposit.contour], -1, 255, -1)
            
            # Boolean mask for indexing
            mask_bool = mask > 0
            
            pixels_rgb = image[mask_bool]
            if len(pixels_rgb) > 0:
                deposit.mean_r = pixels_rgb[:, 0].mean() / 255.0
                deposit.mean_g = pixels_rgb[:, 1].mean() / 255.0
                deposit.mean_b = pixels_rgb[:, 2].mean() / 255.0
            
            pixels_hls = hls[mask_bool]
            if len(pixels_hls) > 0:
                deposit.mean_hue = pixels_hls[:, 0].mean() * 2
                deposit.mean_lightness = pixels_hls[:, 1].mean() / 255.0
                deposit.mean_saturation = pixels_hls[:, 2].mean() / 255.0
            
            deposit.iod = deposit.area * (1 - deposit.mean_lightness)
        
        return deposits
    
    def area_to_um2(self, area_pixels: float) -> float:
        """Convert area from pixels to square micrometers."""
        return area_pixels / (self.pixels_per_um ** 2)
    
    def to_feature_dict(self, deposit: Deposit) -> dict:
        """Convert deposit to feature dictionary."""
        return {
            'id': deposit.id,
            'x': deposit.centroid[0],
            'y': deposit.centroid[1],
            'width': deposit.width,
            'height': deposit.height,
            'area_px': deposit.area,
            'area_um2': self.area_to_um2(deposit.area),
            'perimeter': deposit.perimeter,
            'circularity': deposit.circularity,
            'aspect_ratio': deposit.aspect_ratio,
            'mean_hue': deposit.mean_hue,
            'mean_saturation': deposit.mean_saturation,
            'mean_lightness': deposit.mean_lightness,
            'mean_r': deposit.mean_r,
            'mean_g': deposit.mean_g,
            'mean_b': deposit.mean_b,
            'iod': deposit.iod,
            'label': deposit.label,
            'confidence': deposit.confidence
        }


# =============================================================================
# pH Estimation from BPB (Bromophenol Blue) Color
# =============================================================================
# BPB color change:
#   - pH < 3.0:    Yellow (Hue ~30-60°)
#   - pH 3.0-4.6:  Transition Green→Blue (Hue ~60-200°)
#   - pH > 4.6:    Blue (Hue ~200-240°)
# =============================================================================

# pH category thresholds (in Hue degrees)
PH_HUE_ACIDIC_MAX = 60       # Below this = acidic (yellow)
PH_HUE_BASIC_MIN = 180       # Above this = basic (blue)

# pH estimation parameters (linear interpolation within BPB range)
PH_MIN = 2.5                  # Estimated pH at pure yellow
PH_MAX = 5.5                  # Estimated pH at pure blue
HUE_AT_PH_MIN = 45            # Hue value corresponding to pH_MIN
HUE_AT_PH_MAX = 220           # Hue value corresponding to pH_MAX


def estimate_ph_category(hue: float) -> str:
    """
    Estimate pH category from BPB hue value.
    
    Args:
        hue: Hue value in degrees (0-360)
        
    Returns:
        'acidic', 'transitional', or 'basic'
    """
    if hue < PH_HUE_ACIDIC_MAX:
        return "acidic"
    elif hue < PH_HUE_BASIC_MIN:
        return "transitional"
    else:
        return "basic"


def estimate_ph_value(hue: float) -> float:
    """
    Estimate numeric pH value from BPB hue.
    
    Uses linear interpolation between yellow (acidic) and blue (basic).
    Note: This is an approximation; actual pH depends on many factors.
    
    Args:
        hue: Hue value in degrees (0-360)
        
    Returns:
        Estimated pH value (typically 2.5-5.5 for BPB range)
    """
    # Clamp hue to valid range
    hue_clamped = max(HUE_AT_PH_MIN, min(HUE_AT_PH_MAX, hue))
    
    # Linear interpolation
    ph = PH_MIN + (hue_clamped - HUE_AT_PH_MIN) * (PH_MAX - PH_MIN) / (HUE_AT_PH_MAX - HUE_AT_PH_MIN)
    
    return round(ph, 2)


def calculate_acidity_index(hue: float) -> float:
    """
    Calculate acidity index from hue (0-1 scale).
    
    Args:
        hue: Hue value in degrees (0-360)
        
    Returns:
        Acidity index: 1.0 = fully acidic (yellow), 0.0 = fully basic (blue)
    """
    # Normalize hue to 0-1 scale (inverted: lower hue = higher acidity)
    hue_clamped = max(HUE_AT_PH_MIN, min(HUE_AT_PH_MAX, hue))
    acidity = 1.0 - (hue_clamped - HUE_AT_PH_MIN) / (HUE_AT_PH_MAX - HUE_AT_PH_MIN)
    
    return round(max(0.0, min(1.0, acidity)), 3)


# Legacy function for backward compatibility
def estimate_ph(hue: float) -> str:
    """Estimate pH category from BPB hue value. (Deprecated: use estimate_ph_category)"""
    return estimate_ph_category(hue)

