# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix
import json

# Project root path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.eegnet import EEGNet
from eeg_dataset import EEGNPZDataset
from augmentations import add_gaussian_noise, time_shift, scale_amplitude

# ---------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, *_ in dataloader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ---------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, *_ in dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    augmentations = [add_gaussian_noise, time_shift, scale_amplitude]

    # Load full dataset
    dataset = EEGNPZDataset(npz_path="data/imagined_binary_all_data.npz", augment=False)

    # Step 1: Split unique subjects (60% train, 20% val, 20% test)
    unique_subjects = np.unique(dataset.subjects)
    train_subjects, temp_subjects = train_test_split(unique_subjects, test_size=0.4, random_state=42)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    # Step 2: Map subject IDs to sample indices
    subject_to_indices = {}
    for idx, subject in enumerate(dataset.subjects):
        subject_to_indices.setdefault(subject, []).append(idx)

    train_indices = [idx for s in train_subjects for idx in subject_to_indices[s]]
    val_indices = [idx for s in val_subjects for idx in subject_to_indices[s]]
    test_indices = [idx for s in test_subjects for idx in subject_to_indices[s]]

    # Step 3: Create dataset subsets
    train_dataset = Subset(
        EEGNPZDataset(npz_path="data/imagined_binary_all_data.npz", augment=True, augmentations=augmentations),
        train_indices
    )
    val_dataset = Subset(
        EEGNPZDataset(npz_path="data/imagined_binary_all_data.npz", augment=False),
        val_indices
    )
    test_dataset = Subset(
        EEGNPZDataset(npz_path="data/imagined_binary_all_data.npz", augment=False),
        test_indices
    )

    # Check label balance
    from collections import Counter
    train_labels = [dataset.labels[i] for i in train_indices]
    val_labels = [dataset.labels[i] for i in val_indices]
    test_labels = [dataset.labels[i] for i in test_indices]

    print("Train label distribution:", Counter(train_labels))
    print("Val label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = EEGNet(num_classes=1, num_channels=64, num_samples=320).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    num_epochs = 50
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "models/eegnet_binary.pth")
            print("✅ Model improved. Saved.")
        else:
            trigger_times += 1
            print(f"⚠️  No improvement for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("⏹️  Early stopping triggered.")
                break

    print("Training complete.")

    # Save training history
    history = {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }
    with open("models/training_eegnet_binary_history.json", "w") as f:
        json.dump(history, f)

    # Final evaluation on test set
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from collections import defaultdict

    # Example lists (populate these during inference)
    all_preds = []
    all_labels = []
    all_subjects = []

    model.eval()
    with torch.no_grad():
        for inputs, labels, subjects in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_subjects.extend(subjects)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)

    # Group results per subject
    subject_metrics = {}
    unique_subjects = np.unique(all_subjects)

    for subject in unique_subjects:
        idxs = np.where(all_subjects == subject)[0]
        subject_y_true = all_labels[idxs]
        subject_y_pred = all_preds[idxs]

        report = classification_report(subject_y_true, subject_y_pred, output_dict=True, zero_division=0)
        subject_metrics[subject] = {
            'accuracy': np.mean(subject_y_true == subject_y_pred),
            'precision_0': report['0']['precision'],
            'recall_0': report['0']['recall'],
            'f1_0': report['0']['f1-score'],
            'precision_1': report['1']['precision'],
            'recall_1': report['1']['recall'],
            'f1_1': report['1']['f1-score'],
        }

    # Print per-subject metrics
    for subj, metrics in subject_metrics.items():
        print(f"\n🧠 Subject {subj}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n📊 Test Set Classification Report (Binary):")
    print(classification_report(all_labels, all_preds, digits=4))
    print("🔍 Test Set Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# ---------------------------
if __name__ == "__main__":
    main()