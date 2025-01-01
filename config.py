from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict, Any

@dataclass
class PreprocessConfig:
    size: Tuple[int, int] = (512, 512)
    normalize: bool = True

class SIFTConfig:
    nfeatures: int = 0
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10.0
    sigma: float = 1.6

class FeatureExtractorConfig:
    sift: Optional[SIFTConfig] = SIFTConfig()