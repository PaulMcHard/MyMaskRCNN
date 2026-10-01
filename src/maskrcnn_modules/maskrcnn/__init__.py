"""Supervised Mask R-CNN wrapped as an anomalib module."""

from maskrcnn_modules.maskrcnn.lightning_model import MaskRCNN
from maskrcnn_modules.maskrcnn.post_processing import DetectionPostProcessor

__all__ = ["DetectionPostProcessor", "MaskRCNN"]
