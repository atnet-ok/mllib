from torchvision import transforms

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import numpy as np

preprocess_gray_image = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

preprocess_color_image = None  # To be implemented

preprocess_sensor_data = None  # To be implemented
