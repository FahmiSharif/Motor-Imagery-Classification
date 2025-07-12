"""
This script merge all the preprocessed data into one npz file
"""
import os
import numpy as np
from pathlib import Path
import argparse

def merge_npz_files(preprocessed_dir, output_file):
    preprocessed_dir = Path(preprocessed_dir)
    all_data = []
    all_labels = []
    subject_ids = []

    npz_files = list(preprocessed_dir.rglob("*_imagined_binary.npz"))
    print(f"Found {len(npz_files)} imagined binary .npz files")

    for npz_path in npz_files:
        try:
            subject_id = npz_path.parent.name  # e.g., sub-01 from /.../sub-01/file.npz
            npz = np.load(npz_path)
            data = npz["data"]
            labels = npz["labels"]

            all_data.append(data)
            all_labels.append(labels)
            subject_ids.extend([subject_id] * data.shape[0])
        except Exception as e:
            print(f"[ERROR] Failed to load {npz_path.name}: {e}")

    if not all_data:
        print("[ERROR] No valid binary .npz files loaded. Exiting.")
        return

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    subject_ids = np.array(subject_ids)

    print(f"Total samples: {all_data.shape[0]} | Shape: {all_data.shape} | Labels: {all_labels.shape} | Subjects: {len(subject_ids)}")

    np.savez(output_file, data=all_data, labels=all_labels, subject_ids=subject_ids)
    print(f"Saved merged dataset to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", type=str, default="./data/preprocessed_binary", help="Directory containing imagined binary .npz files")
    parser.add_argument("--output_file", type=str, default="./data/imagined_binary_all_data.npz", help="Output merged .npz file path")
    args = parser.parse_args()

    merge_npz_files(args.preprocessed_dir, args.output_file)