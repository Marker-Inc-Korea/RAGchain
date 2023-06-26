from constants import EMBEDDED_FILES_CACHE_DIRECTORY
import pickle
import os
from datetime import datetime


class EmbeddedFilesCache:
    """
    Not good for asynchronous process
    Only use this for single process because I/O
    """

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

    def get_all_files(self):
        return list(self.cache.keys())

    @classmethod
    def delete_files(cls):
        os.remove(EMBEDDED_FILES_CACHE_DIRECTORY)
