import json
import random

class DataIndex:
    def __init__(self, metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # dict: activity_name → list of file paths
        self.activities = metadata["activities"]

        # flatten all file paths
        self.all_files = []
        for files in self.activities.values():
            self.all_files.extend(files)

    def get_activity_names(self):
        return list(self.activities.keys())

    def get_files_for(self, activity):
        return self.activities[activity]

    def all_file_list(self):
        return self.all_files

    def train_test_split(self, ratio=0.8, shuffle=True):
        all_files = self.all_files.copy()
        if shuffle:
            random.shuffle(all_files)

        n_train = int(len(all_files) * ratio)
        train_files = all_files[:n_train]
        test_files = all_files[n_train:]
        return train_files, test_files
