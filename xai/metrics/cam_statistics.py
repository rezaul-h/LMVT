import numpy as np
from skimage import measure, feature

def activation_area(cam, threshold=0.5):
    """
    Fraction of pixels in cam above threshold.
    """
    return np.mean(cam > threshold)

def cam_noise_ratio(cam, threshold=0.2, min_size=10):
    """
    Ratio of small connected components (area < min_size) to total components.
    """
    mask = cam > threshold
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)
    total = len(props)
    if total == 0:
        return 0.0
    small = sum(1 for p in props if p.area < min_size)
    return small / total

def edge_density(cam):
    """
    Edge density computed via Canny edge detector.
    """
    edges = feature.canny(cam)
    return edges.sum() / edges.size
