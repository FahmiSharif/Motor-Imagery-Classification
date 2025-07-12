import os
import mne
import numpy as np

# === CONFIG === #
edf_root_dir = "./data/raw"
motor_runs = ["R04", "R06", "R08", "R10", "R12", "R14"]
n_subjects = 109

def zscore(data):
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    return (data - mean) / std

def relabel_event(event_code, run_name):
    if run_name in ["R04", "R08", "R12"]:
        return 1 if event_code == 2 else 2
    elif run_name in ["R06", "R10", "R14"]:
        return 3 if event_code == 2 else 4
    else:
        return 0

# Standardize EDF montage naming
channel_rename_map = {
    'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
    'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
    'OZ': 'Oz', 'IZ': 'Iz'
}

save_dir = "./data/preprocessed_MI"
os.makedirs(save_dir, exist_ok=True)

for i in range(1, n_subjects + 1):
    subject_id = f"S{i:03d}"
    subject_dir = os.path.join(edf_root_dir, subject_id)
    subject_data, subject_labels = [], []

    for run in motor_runs:
        edf_filename = f"{subject_id}{run}.edf"
        edf_path = os.path.join(subject_dir, edf_filename)

        if not os.path.exists(edf_path):
            print(f"❌ Missing: {edf_path}")
            continue

        print(f"✅ Processing {edf_filename} ...")

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)

            # Standardize and fix channel names to match montage
            raw.rename_channels(lambda name: name.strip('.').upper())
            raw.rename_channels(channel_rename_map)
            raw.set_montage("standard_1020", on_missing='warn')

            raw.filter(8., 30., fir_design='firwin')

            events, event_id = mne.events_from_annotations(raw)
            keep_event_ids = [2, 3]
            filtered_events = events[np.isin(events[:, 2], keep_event_ids)]
            final_labels = np.array([relabel_event(e[2], run) for e in filtered_events])

            if len(filtered_events) == 0:
                print(f"⚠️ No valid events in {edf_filename}")
                continue

            epochs = mne.Epochs(
                raw,
                filtered_events,
                event_id=None,
                tmin=-0.5,
                tmax=4.0,
                baseline=(-0.5, 0),
                preload=True
            )
            X = zscore(epochs.get_data())
            subject_data.append(X)
            subject_labels.append(final_labels)

        except Exception as e:
            print(f"⚠️ Error processing {edf_filename}: {e}")
            continue

    if subject_data:
        X = np.concatenate(subject_data, axis=0)
        y = np.concatenate(subject_labels, axis=0)
        np.savez_compressed(os.path.join(save_dir, f"{subject_id}_processed.npz"), X=X, y=y)
        print(f"✅ Saved {subject_id} with shape {X.shape} and labels {np.unique(y, return_counts=True)}\n")
    else:
        print(f"❌ No data processed for {subject_id}\n")