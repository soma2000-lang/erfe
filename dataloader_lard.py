import os
import numpy as np
import torch
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class RunwayDataset(Dataset):
    """
    Dataset for runway segmentation from corner coordinates.
    """
  
    def __init__(self, image_paths, coordinates_csv_path, input_shape=(3, 640, 360), num_seg_classes=1, augment=False):
        self.image_paths = image_paths
        self.coordinates_df = pd.read_csv(coordinates_csv_path, sep=';')
        self.input_shape = input_shape
        self.num_seg_classes = num_seg_classes
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)
    
    def create_mask_from_coordinates(self, image_shape, coordinates, class_id=1):
        """Create binary mask from corner coordinates"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        coords = np.array(coordinates, dtype=np.int32)
        

        cv2.fillPoly(mask, [coords], class_id)
        
        return mask
  
    def __getitem__(self, idx):
     
        img_path = self.image_paths[idx] #per image "/home/AD/smajumder/lard/data/photo2.jpeg"
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = os.path.basename(img_path).split('.')[0] #photo2 from the image directory
      
        
    
       
      
        
        
       
        try:
            row = self.coordinates_df.iloc[idx]
            
           
            img= os.path.basename(row['image']).split('.')[0] # from the .csv file
            # h, w = row['height'],row['weight']
            
            coordinates = [
                    [int(row['x_A']), int(row['y_A'])],
                    [int(row['x_B']), int(row['y_B'])],
                    [int(row['x_C']), int(row['y_C'])],
                    [int(row['x_D']), int(row['y_D'])]
                ]
   
                
               
            mask = self.create_mask_from_coordinates((row['height'],row['width']), coordinates, class_id=1)
            
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not find coordinates for {img}. Error: {e}")
            mask = np.zeros((row['height'],row['width']), dtype=np.uint8)
            coordinates = [[0, 0], [0, 0], [0, 0], [0, 0]]
        
        # Apply augmentations
        if self.augment:
            # First apply preprocessing augmentations (image only)
            preprocessing = A.Compose([
                # A.ToGray(p=0.5),
                # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
                # A.MedianBlur(blur_limit=3, p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            ])
            
            preprocessed = preprocessing(image=image)
            image = preprocessed["image"]
            
            # Then apply geometric and other augmentations to both image and mask
            augmentation = A.Compose([
                A.Resize(height=512, width=512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.ShiftScaleRotate(
                #     shift_limit=0.05, 
                #     scale_limit=0.1, 
                #     rotate_limit=15, 
                #     border_mode=cv2.BORDER_CONSTANT, 
                #     p=0.7
                # ),
                # A.Perspective(scale=(0.05, 0.1), p=0.5),
                # A.RandomBrightnessContrast(
                #     brightness_limit=0.15,
                #     contrast_limit=0.15, 
                #     p=0.7
                # ),
                # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                # A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                # A.RandomShadow(
                #     shadow_roi=(0, 0, 1, 1), 
                #     num_shadows_lower=1, 
                #     num_shadows_upper=2, 
                #     shadow_dimension=5, 
                #     p=0.3
                # ),
                # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
                # A.GridDropout(
                #     ratio=0.1, 
                #     unit_size_min=10, 
                #     unit_size_max=40, 
                #     holes_number_x=4, 
                #     holes_number_y=4, 
                #     random_offset=True, 
                #     p=0.2
                # ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            augmented = augmentation(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # If no augmentation, just resize, normalize and convert to tensor
            transform = A.Compose([
                A.Resize(height=512, width=512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            transformed = transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
 
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        

        seg_mask = mask.float() 
        return {
            'image': image, 
            'mask': mask, #image_dir mask
            'coordinates': coordinates, #co-ordinates from the csv file 
            'seg_mask': seg_mask, # mask from the csv file
            'name': img_name
        }
    # you need the mask from the c-ordiates
    # mask from the image dir
    # co-ordinates