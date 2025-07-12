"""
This script performs EEG data preprocessing
"""

import os
import mne
import numpy as np
import argparse
from pathlib import Path
import re

def is_imagined_run(run_id):
    return int(run_id) % 2 == 0 and int(run_id) in [4, 8, 12]

def get_label_mapping_binary(run_id):
    if is_imagined_run(run_id):
        return {"T1": 0, "T2": 1}  # Left = 0, Right = 1
    return {}  # Ignore everything else

def extract_run_number(filename):
    match = re.search(r'R(\d+)', filename)
    if match:
        return match.group(1)
    return "0"

def preprocess_edf(edf_path, save_dir, l_freq=1., h_freq=40., epoch_tmin=0.0, epoch_tmax=2.0, fixed_length=320, do_ica=True):
    run_number = extract_run_number(edf_path.name)
    if not is_imagined_run(run_number):
        print(f"Skipping {edf_path.name}: not an imagined run.")
        return

    raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    raw.set_eeg_reference('average', projection=False)

    if do_ica:
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter='auto')
        ica.fit(raw)
        raw = ica.apply(raw)

    events, event_id = mne.events_from_annotations(raw)
    print(f"Found {len(events)} events in {edf_path.name} | Event IDs: {event_id}")

    label_map = get_label_mapping_binary(run_number)
    if not label_map:
        print(f"[WARNING] No binary labels for {edf_path.name}, skipping.")
        return

    known_events = {k: v for k, v in event_id.items() if k in label_map}
    if not known_events:
        print(f"[WARNING] No known binary events in {edf_path.name}, skipping.")
        return

    epochs = mne.Epochs(
        raw,
        events,
        event_id=known_events,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    data = epochs.get_data()
    raw_labels = epochs.events[:, -1]

    reverse_event_id = {v: k for k, v in known_events.items()}
    mapped_labels = []
    valid_indices = []

    for i, r in enumerate(raw_labels):
        label_name = reverse_event_id.get(r)
        mapped_value = label_map.get(label_name, -1)
        if mapped_value != -1:
            mapped_labels.append(mapped_value)
            valid_indices.append(i)

    if not mapped_labels:
        print(f"[WARNING] No valid binary labels found in {edf_path.name}, skipping.")
        return

    data = data[valid_indices]
    labels = np.array(mapped_labels)

    if data.shape[2] > fixed_length:
        data = data[:, :, :fixed_length]
    elif data.shape[2] < fixed_length:
        pad_width = fixed_length - data.shape[2]
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    data = (data - mean) / (std + 1e-6)

    save_path = Path(save_dir) / (edf_path.stem + "_imagined_binary.npz")
    np.savez(save_path, data=data, labels=labels)
    print(f"Saved binary imagined data to: {save_path}")

def batch_preprocess(data_dir, save_dir):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    edf_files = list(data_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in {data_dir}")

    for edf_file in edf_files:
        try:
            subject_id = edf_file.parent.name
            subject_save_dir = save_dir / subject_id
            subject_save_dir.mkdir(parents=True, exist_ok=True)
            preprocess_edf(edf_file, subject_save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {edf_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to raw EDF files")
    parser.add_argument("--save_dir", type=str, default="./data/preprocessed_binary", help="Where to save .npz files")
    args = parser.parse_args()

    batch_preprocess(args.data_dir, args.save_dir)