import os
import shutil
import random

# Point to your main data directories
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# The IEEE standard split ratio
# The IEEE standard split ratio (Moving OUT of Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def auto_split_data():
    # Find all the category folders you made in data/test (fog, ice, soiling, water)
    categories = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    print("🚀 Starting auto-split process (Extracting from Test folder)...")
    
    for category in categories:
        train_cat_dir = os.path.join(TRAIN_DIR, category)
        val_cat_dir = os.path.join(VAL_DIR, category)
        test_cat_dir = os.path.join(TEST_DIR, category)
        
        # Ensure the train and val target folders exist
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(val_cat_dir, exist_ok=True)
        
        # Grab all the images currently sitting in your test folder
        images = [f for f in os.listdir(test_cat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        
        if total_images == 0:
            print(f"⚠️ Skipping '{category}' (No images found in {test_cat_dir})")
            continue
            
        # Randomly shuffle them
        random.shuffle(images)
        
        # Calculate exactly how many files equal 70% and 15%
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        
        # Slice the shuffled list
        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]
        
        # Physically move the files to the Train folder
        for img in train_images:
            shutil.move(os.path.join(test_cat_dir, img), os.path.join(train_cat_dir, img))
            
        # Physically move the files to the Val folder
        for img in val_images:
            shutil.move(os.path.join(test_cat_dir, img), os.path.join(val_cat_dir, img))
            
        test_left = total_images - train_count - val_count
        print(f"✅ {category.upper()}: Moved {train_count} to Train | Moved {val_count} to Val | Kept {test_left} in Test")

    print("🎉 All done! Your dataset has been perfectly distributed backward from the test vault.")

if __name__ == "__main__":
    auto_split_data()