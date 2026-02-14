from dataclasses import dataclass, asdict
from typing import Optional, Tuple

@dataclass
class PipelineConfig:
    apply_clahe: bool = True
    clip_limit: float = 2.0
    
    apply_denoise: bool = True
    denoise_strength: int = 10
    
    apply_gaussian: bool = False
    gaussian_ksize: int = 5
    
    apply_sharpen: bool = False
    sharpen_strength: float = 1.0
    
    apply_bilateral: bool = False
    
    apply_threshold: bool = True
    block_size: int = 11
    c: int = 2
    
    apply_canny: bool = False
    canny_low: int = 50
    canny_high: int = 150
    
    apply_morph: bool = False
    morph_kernel_size: int = 5
    morph_iterations: int = 1
    
    gamma: float = 1.0
    brightness: int = 0
    contrast: float = 1.0
    
    target_size: Optional[Tuple[int, int]] = None  # (width, height)
    roi_coords: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
