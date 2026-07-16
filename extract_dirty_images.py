import os
import shutil

# Point this to the WoodScape dataset you downloaded
SOURCE_RGB_DIR = "/Users/ojasdesai/Downloads/soiling_dataset/rgbImages"
SOURCE_MASK_DIR = "/Users/ojasdesai/Downloads/soiling_dataset/gtLabels"

# Where we will dump the dirty images for you to look at
OUTPUT_DIR = "data/unsorted_dirty_images"

def extract_dirty_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SOURCE_RGB_DIR) or not os.path.exists(SOURCE_MASK_DIR):
        print("⚠️ Double-check your paths! One of the source folders wasn't found.")
        return
        
    mask_files = [f for f in os.listdir(SOURCE_MASK_DIR) if f.endswith('.png')]
    print(f"Found {len(mask_files)} dirty mask files in gtLabels. Processing...")
    
    # Peek at the first file name to figure out the extension type automatically
    sample_rgb_files = os.listdir(SOURCE_RGB_DIR)
    rgb_ext = ".jpg" # default guess
    if sample_rgb_files:
        # Check if the dataset is using .png or .jpg for the raw images
        for f in sample_rgb_files:
            if f.endswith('.png'):
                rgb_ext = '.png'
                break
            elif f.endswith('.jpg') or f.endswith('.jpeg'):
                rgb_ext = '.jpg'
                break

    count = 0
    for mask_name in mask_files:
        # Strip the mask extension to look for the matching image base name
        base_name = os.path.splitext(mask_name)[0]
        
        # WoodScape fix: sometimes masks end in '_soiling', strip it if needed
        if base_name.endswith('_soiling'):
            base_name = base_name.replace('_soiling', '')
            
        # Try both common naming options
        possible_names = [
            f"{base_name}{rgb_ext}", 
            mask_name.replace('.png', rgb_ext)
        ]
        
        for rgb_name in possible_names:
            src_rgb_path = os.path.join(SOURCE_RGB_DIR, rgb_name)
            dest_rgb_path = os.path.join(OUTPUT_DIR, rgb_name)
            
            if os.path.exists(src_rgb_path):
                shutil.copy(src_rgb_path, dest_rgb_path)
                count += 1
                break # Move to the next mask file once matched
            
    print(f"✅ Successfully copied {count} dirty images to {OUTPUT_DIR}!")

if __name__ == "__main__":
    extract_dirty_frames()