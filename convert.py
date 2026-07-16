import tensorflow as tf
import os
import subprocess
import shutil

print("⏳ Loading original Keras model...")
model = tf.keras.models.load_model('soiling_severity_model.keras', compile=False)

print("🏗️ Rebuilding architecture safely (avoiding topology destruction)...")
clean_layers = []

# Iterate through the top-level layers
for layer in model.layers:
    cls_name = layer.__class__.__name__.lower()
    
    # 1. Check if the augmentations are hiding inside a nested Sequential block
    if isinstance(layer, tf.keras.Sequential):
        has_aug = any(
            any(x in sub.__class__.__name__.lower() for x in ['random', 'flip', 'rotation', 'zoom', 'contrast', 'crop']) 
            for sub in layer.layers
        )
        if has_aug:
            print(f"✂️ Safely dropping nested augmentation block: {layer.name}")
            continue # Skip the bad block entirely
        else:
            print(f"✅ Keeping clean nested block: {layer.name}")
            clean_layers.append(layer)
            
    # 2. Check if it's a standalone augmentation layer
    elif any(x in cls_name for x in ['random', 'flip', 'rotation', 'zoom', 'contrast', 'crop']):
        print(f"✂️ Safely dropping standalone augmentation layer: {layer.name}")
        continue
        
    # 3. Keep all core neural network layers (Dense, Conv2D, pre-trained base models)
    else:
        print(f"✅ Keeping core architecture layer: {layer.name} ({layer.__class__.__name__})")
        clean_layers.append(layer)

# Build the new, pristine shell model
clean_model = tf.keras.Sequential(clean_layers)
# Ensure the input shape exactly matches your 224x224 RGB pipeline
clean_model.build((None, 224, 224, 3))

print("🧠 Executing weight transplant (0 loss of logic!)...")
# Because augmentation layers have 0 parameters, the flat weight lists align perfectly.
clean_model.set_weights(model.get_weights())

print("📦 Exporting pristine inference graph...")
temp_saved_model_dir = "tmp_saved_model"
if os.path.exists(temp_saved_model_dir):
    shutil.rmtree(temp_saved_model_dir)

# Use Keras 3's official exporter on the clean shell
clean_model.export(temp_saved_model_dir)

print("🔄 Converting pure graph to ONNX format...")
output_onnx_path = "soiling_severity_model.onnx"
command = f"python -m tf2onnx.convert --saved-model {temp_saved_model_dir} --output {output_onnx_path} --opset 13"
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Clean up
if os.path.exists(temp_saved_model_dir):
    shutil.rmtree(temp_saved_model_dir)

if result.returncode == 0:
    print(f"🎉 Success! Bulletproof ONNX model saved as: {output_onnx_path}")
else:
    print("❌ ONNX Conversion failed. Error log below:")
    print(result.stderr)