import os
import shutil
import random

# ==========================================================
# 1. DEFINE PATHS AND DATA SPLIT RATIOS
# ==========================================================
# Source folder containing your raw, unsorted clean images
SOURCE_CLEAN_DIR = "data/clean"

# Target directories where the clean data will be distributed
TRAIN_TARGET_DIR = "data/train/clean"
VAL_TARGET_DIR = "data/val/clean"
TEST_TARGET_DIR = "data/test/clean"

# Strict IEEE standard split ratios (70% train, 15% validation, 15% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def split_clean_dataset():
    print("🚀 Starting the clean image auto-split process...")
    
    # Check if the source folder actually exists before doing anything
    if not os.path.exists(SOURCE_CLEAN_DIR):
        print(f"⚠️ Error: The folder '{SOURCE_CLEAN_DIR}' was not found. Please verify the folder name!")
        return

    # Automatically build the empty target 'clean' subfolders if they don't exist yet
    os.makedirs(TRAIN_TARGET_DIR, exist_ok=True)
    os.makedirs(VAL_TARGET_DIR, exist_ok=True)
    os.makedirs(TEST_TARGET_DIR, exist_ok=True)

    # Gather all valid image files inside the main clean directory
    images = [f for f in os.listdir(SOURCE_CLEAN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)
    
    if total_images == 0:
        print(f"⚠️ No images found inside '{SOURCE_CLEAN_DIR}'. Drop your clean camera files there first!")
        return
        
    print(f"📊 Found {total_images} clean baseline frames. Randomizing distribution...")
    
    # Shuffle the dataset randomly to guarantee a completely unbiased distribution
    random.shuffle(images)
    
    # Calculate exact target image numbers based on the math splits
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    
    # Slice the randomized file array for each directory tier
    train_images = images[:train_count]
    val_images = images[train_count : train_count + val_count]
    test_images = images[train_count + val_count :] # Takes whatever is left over to protect against rounding issues
    
    # ==========================================================
    # 2. EXECUTE THE PHYSICAL FILE MOVES
    # ==========================================================
    # Move training chunk
    for img in train_images:
        shutil.move(os.path.join(SOURCE_CLEAN_DIR, img), os.path.join(TRAIN_TARGET_DIR, img))
        
    # Move validation chunk
    for img in val_images:
        shutil.move(os.path.join(SOURCE_CLEAN_DIR, img), os.path.join(VAL_TARGET_DIR, img))
        
    # Move testing chunk
    for img in test_images:
        shutil.move(os.path.join(SOURCE_CLEAN_DIR, img), os.path.join(TEST_TARGET_DIR, img))
        
    print("🎉 Dataset split successfully finished!")
    print(f"📁 Moved {len(train_images)} images -> {TRAIN_TARGET_DIR}")
    print(f"📁 Moved {len(val_images)} images -> {VAL_TARGET_DIR}")
    print(f"📁 Moved {len(test_images)} images -> {TEST_TARGET_DIR}")

if __name__ == "__main__":
    split_clean_dataset()