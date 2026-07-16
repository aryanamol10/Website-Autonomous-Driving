# Running/testing: 
# 1. python train.py
# 2. streamlit run app.py
# or python -m streamlit run app.py if that doesn't work

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import json
from sklearn.metrics import confusion_matrix

cudnn.benchmark = True

# ==========================================
# 1. LOAD DATA & FOLDER STRUCTURE
# ==========================================
# Data augmentation and normalization for training
# Just normalization for validation and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # ADDED: Strict Test Set evaluation transforms
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Pointing directly to your local 'data' folder
data_dir = 'data'

# ADDED 'test' to all data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

# Automatically extracts your 4 categories (fog, ice, soiling, water)
class_names = image_datasets['train'].classes
print(f"Detecting {len(class_names)} categories: {class_names}")

# Device configuration (Checks for Apple Silicon Mac GPU first, then Nvidia, then CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Training on device: {device}")

# ==========================================
# 2. TRAINING LOOP (From Tutorial)
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Saving to your project folder instead of a temporary directory
    best_model_params_path = 'best_model_copy.pt'
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            # Track metrics for the Streamlit UI
            if phase == 'train':
                history['train_loss'].append(float(epoch_loss))
                history['train_acc'].append(float(epoch_acc))
            elif phase == 'val':
                history['val_loss'].append(float(epoch_loss))
                history['val_acc'].append(float(epoch_acc))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
                print(f"*** New best model saved with accuracy: {best_acc:.4f} ***")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    
    # Return both the model and the training history for the next step
    return model, history

# ==========================================
# 3. COMPLETE DATASET EVALUATION (Train, Val, Test)
# ==========================================
def evaluate_all_sets(model, history):
    print("\n🚀 Generating Confusion Matrices for Train, Val, and Test sets...")
    model.eval()
    
    # We will loop through all three splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        all_preds = []
        all_labels = []
        
        # Turn off gradients for pure evaluation mapping
        for inputs, labels in dataloaders[split]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.data.cpu().numpy())
            
        # Calculate dynamic matrix for current split
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names))).tolist()
        
        # Save it directly into our history dictionary
        history[f'{split}_confusion_matrix'] = cm
        
        if split == 'test':
            test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
            history['test_acc'] = float(test_acc)
            print(f'🔥 Final True Test Accuracy: {test_acc*100:.2f}%')
    
    # Write the expanded matrix dictionary out to disk as a copy to prevent overwriting
    with open('metrics_copy.json', 'w') as f:
        json.dump(history, f)
    print("✅ metrics_copy.json successfully updated with all 3 Confusion Matrices!")

# ==========================================
# 4. FINETUNING SETUP & EXECUTION
# ==========================================
if __name__ == '__main__':
    # CONNECTED TO CUSTOM PRETRAINING:
    # Instead of pulling from online torchvision libraries, we import our handwritten model setup module
    from custom_pretraining import get_custom_pretrained_model
    
    # Instantiate the backbone from our handwritten custom architecture script
    model_ft = get_custom_pretrained_model(len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, model_history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    
    # Run the expanded matrix generator script
    evaluate_all_sets(model_ft, model_history)
