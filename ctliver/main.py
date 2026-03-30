import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from model import HCCModel
from data_loader import HCCDataset
from train_eval import train, evaluate
from gradcam import generate_gradcam
from report_utils import generate_latex_table

# 🔒 Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create results folder
os.makedirs("results", exist_ok=True)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
csv_file = "ctliver/hcc-data-complete-balanced.csv"
img_dir = "ctliver/dataset/div-images"

# First, load full dataset to get labels for stratified split
full_dataset = HCCDataset(csv_file=csv_file, img_dir=img_dir, augment=False)
labels = full_dataset.labels

# Stratified train/test split (80/20)
train_idx, test_idx = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    random_state=SEED, 
    stratify=labels
)

# Create train and test datasets with augmentation for training
train_dataset = HCCDataset(csv_file=csv_file, img_dir=img_dir, augment=True)
test_dataset = HCCDataset(csv_file=csv_file, img_dir=img_dir, augment=False)

train_set = Subset(train_dataset, train_idx)
test_set = Subset(test_dataset, test_idx)

# Print class distribution
train_labels = [labels[i] for i in train_idx]
test_labels = [labels[i] for i in test_idx]
print(f"Train set: {Counter(train_labels)}")
print(f"Test set: {Counter(test_labels)}")

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1)

# Calculate class weights for imbalanced data
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])], dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# Model - use ResNet18 with pretrained weights
model = HCCModel(num_classes=2, pretrained=True).to(device)

# Lower learning rate for fine-tuning pretrained model
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Loss with class weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Learning rate scheduler
num_epochs = 30
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)

# Train for more epochs
best_loss = float('inf')
for epoch in range(num_epochs):
    loss = train(model, train_loader, criterion, optimizer, device, scheduler)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Save best model
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), "results/best_model.pth")

# Load best model for evaluation
model.load_state_dict(torch.load("results/best_model.pth"))

# Evaluate
report = evaluate(model, test_loader, device)
print("Classification Report:", report)

# --- Visualize some test images with predictions ---
import matplotlib.pyplot as plt

def visualize_predictions(model, test_loader, device, class_names, num_images=8):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for clin, labels in test_loader:
            clin, labels = clin.to(device), labels.to(device)
            outputs = model(clin)
            preds = torch.argmax(outputs, dim=1)

            for i in range(clin.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.savefig("results/sample_predictions.png")
                    plt.show()
                    return
                
                img = clin[i].cpu().permute(1, 2, 0).numpy()

                plt.subplot(2, num_images//2, images_shown+1)
                plt.imshow(img, cmap="gray")
                plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}")
                plt.axis("off")

                images_shown += 1

# Call function
class_names = ["Healthy", "Disease"]
visualize_predictions(model, test_loader, device, class_names, num_images=8)

# --- Export LaTeX table ---
generate_latex_table(report)

print("\n✅ Training complete! Check results/ folder for outputs.")
