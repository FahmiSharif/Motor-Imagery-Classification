'''
This script train and test EEGNet on all subjects, and save all results + history in one JSON.
All the feature extraction happened here.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay

# Local imports (adjust paths if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eegnet import EEGNet
from eeg_dataset import EEGNPZDataset
from augmentations import add_gaussian_noise, time_shift, scale_amplitude

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels, *_ in dataloader:
        inputs, labels = inputs.float().to(device), labels.float().to(device)
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    total_inference_time = 0.0
    with torch.no_grad():
        for inputs, labels, *_ in dataloader:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            start_time = time.time()
            outputs = model(inputs).squeeze(1)
            total_inference_time += time.time() - start_time
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
    avg_inference_time = total_inference_time / total if total > 0 else 0
    return total_loss / total, correct / total, all_preds, all_labels, avg_inference_time

def run_all_subjects():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    npz_path = "data/imagined_binary_all_data.npz"
    augmentations = [add_gaussian_noise, time_shift, scale_amplitude]
    full_dataset = EEGNPZDataset(npz_path=npz_path, augment=False)
    #unique_subjects = np.unique(full_dataset.subjects)
    unique_subjects = ['S001']

    subject_to_indices = defaultdict(list)
    for idx, subj in enumerate(full_dataset.subjects):
        subject_to_indices[subj].append(idx)

    all_results = {}

    for subject in unique_subjects:
        print(f"\n🔁 Testing Subject {subject}...")

        test_indices = subject_to_indices[subject]
        train_indices = [idx for subj, indices in subject_to_indices.items() if subj != subject for idx in indices]

        train_labels = [full_dataset.labels[i] for i in train_indices]
        test_labels = [full_dataset.labels[i] for i in test_indices]

        if len(set(test_labels)) < 2:
            print(f"⚠️ Skipping subject {subject} due to only one class in test set.")
            continue

        train_dataset = Subset(
            EEGNPZDataset(npz_path=npz_path, augment=True, augmentations=augmentations), train_indices)
        test_dataset = Subset(
            EEGNPZDataset(npz_path=npz_path, augment=False), test_indices)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = EEGNet(num_classes=1, num_channels=64, num_samples=320).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc, patience, trigger_times = 0, 3, 0
        losses, accs = [], []

        for epoch in range(1):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            losses.append(train_loss)
            accs.append(train_acc)
            if epoch % 5 == 0 or trigger_times == patience - 1:
                print(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
            if train_acc > best_acc:
                best_acc = train_acc
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        _, test_acc, y_pred, y_true, inference_time = evaluate(model, test_loader, criterion, device)
        print("y_true:", y_true[:10], type(y_true[0]))
        print("y_pred:", y_pred[:10], type(y_pred[0]))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        print("Report keys:", report.keys())
        cm = confusion_matrix(y_true, y_pred)

        print(f"✅ Subject {subject} Accuracy: {test_acc:.4f}")
        print(f"📉 Confusion Matrix:\n{cm}")
        print(f"⏱️ Inference Time: {inference_time:.6f}s")

        all_results[subject] = {
            "accuracy": test_acc,
            "inference_time_sec": inference_time,
            "loss": losses,
            "accuracy_history": accs,
            "precision_0": report.get('0', {}).get("precision", 0.0),
            "recall_0": report.get('0', {}).get("recall", 0.0),
            "f1_0": report.get('0', {}).get("f1-score", 0.0),
            "precision_1": report.get('1', {}).get("precision", 0.0),
            "recall_1": report.get('1', {}).get("recall", 0.0),
            "f1_1": report.get('1', {}).get("f1-score", 0.0),
            "confusion_matrix": cm.tolist()
        }

    os.makedirs("models", exist_ok=True)
    with open("models/all_subjects_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    run_all_subjects()