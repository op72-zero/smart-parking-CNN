# scripts/extract_slots.py
import os
import cv2
import shutil
from tqdm import tqdm

print("🚀 EXTRACTING INDIVIDUAL PARKING SLOTS...")
print("=" * 50)

# Create output folders
os.makedirs("data/raw/empty", exist_ok=True)
os.makedirs("data/raw/occupied", exist_ok=True)

# Counters
empty_count = 0
occupied_count = 0

def extract_slots_from_split(split_name):
    """Extract parking slots from train/valid/test splits"""
    global empty_count, occupied_count
    
    images_dir = f"PKLot/{split_name}/images"
    labels_dir = f"PKLot/{split_name}/labels"
    
    if not os.path.exists(images_dir):
        print(f"❌ {images_dir} not found!")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n📁 Processing {split_name} set ({len(image_files)} images)...")
    
    for image_file in tqdm(image_files, desc=f"Extracting {split_name}"):
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue  # Skip if no label file
            
        # Load the full parking lot image
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Read YOLO format labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Extract each parking slot from this image
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:  # YOLO format: class x_center y_center width height
                class_id = int(parts[0])  # 0=empty, 1=occupied
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Calculate bounding box coordinates
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)
                
                # Crop the individual parking slot
                slot_img = img[y1:y2, x1:x2]
                
                if slot_img.size == 0:
                    continue  # Skip empty crops
                
                # Resize to standard size for CNN (224x224 pixels)
                slot_img = cv2.resize(slot_img, (224, 224))
                
                # Save based on class
                if class_id == 0:  # Empty parking slot
                    output_file = f"data/raw/empty/empty_{empty_count:06d}.jpg"
                    cv2.imwrite(output_file, slot_img)
                    empty_count += 1
                else:  # Occupied parking slot (class_id == 1)
                    output_file = f"data/raw/occupied/occupied_{occupied_count:06d}.jpg"
                    cv2.imwrite(output_file, slot_img)
                    occupied_count += 1

# Process all dataset splits
for split in ['train', 'valid', 'test']:
    extract_slots_from_split(split)

print(f"\n🎉 EXTRACTION COMPLETE!")
print(f"📸 Empty parking slots: {empty_count} images")
print(f"📸 Occupied parking slots: {occupied_count} images")
print(f"📊 Total individual slots: {empty_count + occupied_count}")
print(f"💾 Saved in: data/raw/empty/ and data/raw/occupied/")

print("\n🚀 NEXT STEP: Run -> python src/data_preprocessing.py")