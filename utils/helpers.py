import time
import numpy as np

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def compute_class_weights(labels, num_classes):
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return class_weights
