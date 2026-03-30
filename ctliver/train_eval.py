import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0
    for clin, label in train_loader:
        clin, label = clin.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(clin)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def evaluate(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for clin, label in val_loader:
            clin, label = clin.to(device), label.to(device)
            output = model(clin)
            pred = torch.argmax(output, dim=1)  
            y_true.extend(label.cpu().numpy())  
            y_pred.extend(pred.cpu().numpy())

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Classification report (dict for return, str for saving)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0,1])
    report_str = classification_report(y_true, y_pred, zero_division=0, labels=[0, 1])

    # Confusion matrix (force labels to [0,1] so it’s always 2x2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix:\n",cm)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Save classification report
    with open("results/classification_report.txt", "w") as f:
        f.write(report_str)

    return report_dict
