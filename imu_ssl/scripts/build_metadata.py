import os
import json
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # imu_ssl/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_JSON = DATA_DIR / "metadata.json"

VALID_EXT = {".hdf5", ".h5"}

# -----------------------------
# Script
# -----------------------------
def scan_data_folder(data_dir):
    metadata = {"activities": {}}

    # loop over each subdirectory inside data/
    for item in sorted(os.listdir(data_dir)):
        activity_path = data_dir / item

        # skip non-directories (e.g., description.txt)
        if not activity_path.is_dir():
            continue

        # scan all hdf5 files inside that directory
        files = []
        for fn in sorted(os.listdir(activity_path)):
            full_path = activity_path / fn
            if full_path.suffix.lower() in VALID_EXT:
                files.append(str(full_path))

        # only add activity if it has hdf5 files
        if len(files) > 0:
            metadata["activities"][item] = files
            print(f"[OK] {item}: {len(files)} files")
        else:
            print(f"[SKIP] {item}: no hdf5 files")

    return metadata


def main():
    print("Scanning data folder:", DATA_DIR)
    metadata = scan_data_folder(DATA_DIR)

    # save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved metadata to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
