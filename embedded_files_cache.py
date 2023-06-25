from constants import EMBEDDED_FILES_CACHE_DIRECTORY
import pickle
import os
from datetime import datetime


class EmbeddedFilesCache:
    def __init__(self):
        if os.path.exists(EMBEDDED_FILES_CACHE_DIRECTORY):
            with open(EMBEDDED_FILES_CACHE_DIRECTORY, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def is_exist(self, full_file_path):
        try:
            _ = self.cache[full_file_path]
            return True
        except:
            return False

    def add(self, full_file_path):
        self.cache[full_file_path] = datetime.now()

    def save(self):
        with open(EMBEDDED_FILES_CACHE_DIRECTORY, 'wb') as f:
            pickle.dump(self.cache, f)
