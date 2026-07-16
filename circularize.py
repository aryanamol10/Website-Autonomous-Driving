import os
from PIL import Image, ImageDraw

# Root data directory containing train, val, and test
DATA_DIR = "data"
PHASES = ["train", "val", "test"]

def apply_circular_mask(image_path):
    """Opens an image, forces it into a centered circular mask with black corners,

    and saves it back in place.
    """
    with Image.open(image_path) as img:
        # Ensure image is in RGB mode (drops alpha channel alpha glitches if any)
        img = img.convert("RGB")
        width, height = img.size
        
        # 1. Create a pure black background of the exact same size
        background = Image.new("RGB", (width, height), (0, 0, 0))
        
        # 2. Create a grayscale mask initialized to pure black (0)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 3. Calculate bounding box for a centered circle using the smaller dimension
        diameter = min(width, height)
        left = (width - diameter) // 2
        top = (height - diameter) // 2
        right = left + diameter
        bottom = top + diameter
        
        # Draw a solid white circle (255) on the mask
        draw.ellipse([left, top, right, bottom], fill=255)
        
        # 4. Blend them: keep original image where mask is white, use black background where mask is black
        circular_img = Image.composite(img, background, mask)
        
        # Save the modified image right back over the old one
        circular_img.save(image_path)

def circularize_dataset():
    print("🌕 Starting to circularize all images to eliminate shape bias...")
    processed_count = 0
    
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Error: '{DATA_DIR}' directory not found. Make sure you are in the right folder!")
        return

    # Loop through data/train, data/val, and data/test
    for phase in PHASES:
        phase_dir = os.path.join(DATA_DIR, phase)
        if not os.path.exists(phase_dir):
            continue
            
        # Loop through categories (clean, fog, ice, soiling, water)
        for category in os.listdir(phase_dir):
            category_dir = os.path.join(phase_dir, category)
            if not os.path.isdir(category_dir):
                continue
                
            print(f"Processing category: {phase}/{category}...")
            
            # Process every single image file inside
            for filename in os.listdir(category_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_dir, filename)
                    try:
                        apply_circular_mask(img_path)
                        processed_count += 1
                    except Exception as e:
                        print(f"❌ Skipped corrupt file {filename}: {e}")
                        
    print(f"\n🎉 Success! Circularized {processed_count} images across all dataset splits.")

if __name__ == "__main__":
    circularize_dataset()