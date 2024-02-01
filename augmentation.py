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
        self.funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        self.funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))   
        
    def set_train_transform(self):
        # self.funcs.append(A.RandomBrightnessContrast(p=0.3))
        # self.funcs.append(A.HueSaturationValue(p=0.3))
        
        #base transform
        self.funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        self.funcs.append(A.Normalize(mean=(0.7667397, 0.77095854, 0.7755067), std=(0.18619478, 0.17846268, 0.1717753)))   
        self.funcs.append(A.CLAHE()),
        self.funcs.append(A.ChannelShuffle(p=0.5))
        self.funcs.append(A.MotionBlur(p=0.5))
        self.funcs.append(A.Normalize(mean=(0.7667397, 0.77095854, 0.7755067), std=(0.18619478, 0.17846268, 0.1717753)))
        # self.funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))   


    def set_valid_transform(self):
        self.funcs.append(A.Normalize(mean=(0.7667397, 0.77095854, 0.7755067), std=(0.18619478, 0.17846268, 0.1717753)))
        # self.funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        
        
        
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
    