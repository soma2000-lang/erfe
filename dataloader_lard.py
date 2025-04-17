import os
import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from config_lard import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE

class RunwayDataset(Dataset):
    """
    Dataset for runway segmentation from corner coordinates.
    """
  
    def __init__(self, image_paths, coordinates_csv_path, input_shape=(3, 1024, 1024), num_seg_classes=NUM_SEG_CLASSES, augment=False):
        self.image_paths = image_paths
        self.coordinates_df = pd.read_csv(coordinates_csv_path, sep=';')
        self.input_shape = input_shape
        self.num_seg_classes = num_seg_classes
        self.augment = augment
        self.img_to_row = {}
        for idx, row in self.coordinates_df.iterrows():
            img_basename = row["image"].split('/')[-1].split(".")[0]
            self.img_to_row[img_basename] = row
        self.valid_image_paths = []
        for img_path in image_paths:
            img_name = img_path.split('/')[-1].split(".")[0]
            if img_name in self.img_to_row:
                row = self.img_to_row[img_name]
                df_img_name = row["image"].split('/')[-1].split(".")[0]
                if img_name == df_img_name:
                    self.valid_image_paths.append(img_path)

    def __len__(self):
        return len(self.valid_image_paths)
    
   
    def pad_triangle_to_quad(self, triangle, t=1):
        A, B, C = triangle
        len_AB = np.linalg.norm(B - A)
        len_AC = np.linalg.norm(C - A)
        len_BC = np.linalg.norm(C - B)
        
        if len_AB <= len_AC and len_AB <= len_BC:
            D = C - np.array([5, 0])  
            trapezium = np.array([A, B, C, D])
        elif len_AC <= len_AB and len_AC <= len_BC:
            D = B - np.array([5, 0])
            trapezium = np.array([A, C, B, D])
        else:
            D = A - np.array([5, 0])
            trapezium = np.array([B, C, A, D])
             
        return trapezium

    def sort_points_clockwise(self, pts):
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return pts[sorted_indices]

    def create_mask_from_coordinates(self, image_shape, coordinates, class_id=1):
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        coords = np.array(coordinates, dtype=np.int32)
       
        if coords.shape[0] == 3:
            coords = self.pad_triangle_to_quad(coords)  
            coords = self.sort_points_clockwise(coords)  
        elif coords.shape[0] == 4:
            coords = self.sort_points_clockwise(coords)
        else:
            print(f"Unexpected number of points: {coords.shape[0]}")

        coords = coords.astype(np.int32)

        
        cv2.fillPoly(mask, [coords], class_id)

        return mask
    
   
    def preprocessaugment_image(self, image, mask):
   

        rescale_transforms = [
            A.LongestMaxSize(1024, p=1.0),
            A.PadIfNeeded(
                min_height=1024, 
                min_width=1024, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=[255, 255, 255], 
                p=1.0
            )
        ]

        aug_transforms = []
        
        if self.augment:
            
            aug_transforms.append(A.HorizontalFlip(p=0.05))

  
        aug_transforms.extend([
            A.Sequential(rescale_transforms, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #A.Normalize(p=1.0),
            ToTensorV2()
        ])

        transform = A.Compose(aug_transforms)

        transformed = transform(image=image, mask=mask)

        return transformed

    def __getitem__(self, idx):
        img_path = self.valid_image_paths[idx] 
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_name = img_path.split('/')[-1].split(".")[0]
        row = self.img_to_row[img_name]
        
        try:
            coordinates = [
                [int(row['x_A']), int(row['y_A'])],
                [int(row['x_B']), int(row['y_B'])],
                [int(row['x_C']), int(row['y_C'])],
                [int(row['x_D']), int(row['y_D'])]
            ]
            
            mask = self.create_mask_from_coordinates((image.shape[0], image.shape[1]), coordinates, class_id=1)
        except (IndexError, KeyError, ValueError) as e:
            print(f"Warning: Could not find coordinates for {img_name}. Error: {e}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            coordinates = [[0, 0], [0, 0], [0, 0], [0, 0]]
        
        transformed = self.preprocessaugment_image(image, mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        seg_mask = mask.float()
        if np.random.rand() < 0.01:
            plt.subplot(1,2,1)
            plt.imshow(image.permute(1,2,0).cpu().numpy())  # Image
            plt.subplot(1,2,2)
            plt.imshow(mask.squeeze(0).cpu().numpy())  # Mask
            plt.show()
            plt.savefig('image_mask.png')

        return {
            'image': image, 
            'coordinates': coordinates,
            'seg_mask': seg_mask,
            'name': img_name
        }