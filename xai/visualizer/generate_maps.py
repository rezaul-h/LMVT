import cv2
import numpy as np
from matplotlib import cm

def overlay_cam_on_image(image, cam, alpha=0.5, colormap='jet'):
    """
    Overlay heatmap cam on original image.
    image: HxWx3 uint8
    cam: HxW float array in [0,1]
    """
    # Create heatmap
    heatmap = cm.get_cmap(colormap)(cam)[:, :, :3]  # RGBA to RGB
    heatmap = np.uint8(heatmap * 255)
    # Overlay
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    return overlay

def save_cam(image_path, cam, output_path, alpha=0.5, colormap='jet'):
    """
    Load image, overlay cam, and save to output_path.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = overlay_cam_on_image(image, cam, alpha, colormap)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)
