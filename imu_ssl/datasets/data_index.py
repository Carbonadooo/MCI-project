import json

class DataIndex:
    def __init__(self, metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.activities = metadata["activities"]

        self.all_files = []
        for files in self.activities.values():
            self.all_files.extend(files)

    def get_activity_names(self):
        return list(self.activities.keys())

    def get_files_for(self, activity):
        return self.activities[activity]

    def all_file_list(self):
        return self.all_files
