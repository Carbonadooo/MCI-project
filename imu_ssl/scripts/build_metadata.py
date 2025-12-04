import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train_set"
VAL_DIR   = DATA_DIR / "validate_set"

VALID_EXT = {".hdf5", ".h5"}

def scan_split(split_dir):
    metadata = {"activities": {}}

    for activity in sorted(os.listdir(split_dir)):
        activity_path = split_dir / activity
        if not activity_path.is_dir():
            continue

        files = [
            str(activity_path / fn)
            for fn in sorted(os.listdir(activity_path))
            if Path(fn).suffix.lower() in VALID_EXT
        ]

        if len(files) > 0:
            metadata["activities"][activity] = files
            print(f"[OK] {activity}: {len(files)} files")

    return metadata


def main():
    print("Building metadata...")

    meta_train = scan_split(TRAIN_DIR)
    meta_val   = scan_split(VAL_DIR)

    with open(DATA_DIR / "metadata_train.json", "w") as f:
        json.dump(meta_train, f, indent=4)

    with open(DATA_DIR / "metadata_val.json", "w") as f:
        json.dump(meta_val, f, indent=4)

    print("\nSaved metadata_train.json and metadata_val.json")


if __name__ == "__main__":
    main()
