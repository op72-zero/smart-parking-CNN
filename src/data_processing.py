import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.img_size = (224, 224)  # Standard size for CNN models
        
    def create_directory_structure(self):
        """Create the necessary directory structure"""
        directories = [
            'splits/train/empty',
            'splits/train/occupied',
            'splits/val/empty',
            'splits/val/occupied',
            'splits/test/empty',
            'splits/test/occupied'
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.processed_data_path, directory), exist_ok=True)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def organize_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """Organize data into train/val/test splits"""
        print("Organizing data into train/val/test splits...")
        
        # Assuming raw data is organized in 'empty' and 'occupied' folders
        empty_dir = os.path.join(self.raw_data_path, 'empty')
        occupied_dir = os.path.join(self.raw_data_path, 'occupied')
        
        # Get all image paths
        empty_images = [os.path.join(empty_dir, f) for f in os.listdir(empty_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        occupied_images = [os.path.join(occupied_dir, f) for f in os.listdir(occupied_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(empty_images)} empty images")
        print(f"Found {len(occupied_images)} occupied images")
        
        # Create labels
        empty_labels = [0] * len(empty_images)  # 0 for empty
        occupied_labels = [1] * len(occupied_images)  # 1 for occupied
        
        # Combine data
        all_images = empty_images + occupied_images
        all_labels = empty_labels + occupied_labels
        
        # Split data: first into train+val and test, then train+val into train and val
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_images, all_labels, test_size=test_size, 
            random_state=random_state, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        # Copy images to respective directories
        self._copy_images(X_train, y_train, 'train')
        self._copy_images(X_val, y_val, 'val')
        self._copy_images(X_test, y_test, 'test')
        
        print("Data organization completed!")
    
    def _copy_images(self, image_paths, labels, split_type):
        """Copy images to their respective directories"""
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            if label == 0:
                target_dir = os.path.join(self.processed_data_path, f'splits/{split_type}/empty')
            else:
                target_dir = os.path.join(self.processed_data_path, f'splits/{split_type}/occupied')
            
            # Copy and preprocess image
            processed_img = self.preprocess_image(img_path)
            if processed_img is not None:
                filename = os.path.basename(img_path)
                # Convert back to uint8 for saving
                save_img = (processed_img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(target_dir, filename), save_img)

if __name__ == "__main__":
    preprocessor = DataPreprocessor('data/raw', 'data/processed')
    preprocessor.create_directory_structure()
    preprocessor.organize_data()