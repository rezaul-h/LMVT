import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


class LungCancerDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with 'image_path' and 'label' columns.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load image path and label
        img_path = self.data.iloc[index]['image_path']
        label = int(self.data.iloc[index]['label'])

        # Read image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label
