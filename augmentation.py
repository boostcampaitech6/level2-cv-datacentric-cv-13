import numpy as np
import cv2
import albumentations as A

class BaseTransform:
    def __init__(self):
        self.funcs = []
    
    def get_transform(self):
        transform = A.Compose(self.funcs)
        return transform
    
    def set_transform(self):
        # self.funcs.append(A.RandomBrightnessContrast(p=0.3))
        # self.funcs.append(A.HueSaturationValue(p=0.3))
        
        #base transform
        self.funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        self.funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))   

class SharpenTransform:
    def __init__(self):
        self.funcs = []
    
    def get_transform(self):
        transform = A.Compose(self.funcs)
        return transform
    
    def set_transform(self):
        # self.funcs.append(A.RandomBrightnessContrast(p=0.3))
        # self.funcs.append(A.HueSaturationValue(p=0.3))
        
        #base transform
        self.funcs.append(A.Sharpen((0.2, 0.5), (0.5, 1), p=0.5))
        self.funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        self.funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
