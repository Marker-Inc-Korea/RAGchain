from RAGchain.utils.linker.base import BaseLinker
from typing import Union
from uuid import UUID

import os
import json


class LocalDBSingleton(BaseLinker):
    """
    LocalDBSingleton is a singleton class that allows the role of a linker
    to be played locally to use JSON file without using an external DB like redis or dynamo.
    """
    def __init__(self):
        json_path = os.getenv("JSON_FILE_PATH")

        if json_path is None:
            raise ValueError("Please set JSON_FILE_PATH to environment variable")

        self.json_path = json_path
        self.create_or_load_json()
        self.data = {}

    def create_json(self):
        with open(self.json_path, "w") as f:
            json.dump({}, f)
        self.data = {}

    def load_json(self):
        try:
            with open(self.json_path, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("JSON file not found")
        except json.decoder.JSONDecodeError:
            raise ValueError("Invalid JSON file")

    def create_or_load_json(self):
        if not os.path.exists(self.json_path):
            self.create_json()
        else:
            self.load_json()

    def put_json(self, id: Union[UUID, str], json_data: dict):
        self.data[id] = json_data
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)

    def get_json(self, ids: list[Union[UUID, str]]):
        return [self.data.get(find_id) for find_id in ids]

    def flush_db(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)
        else:
            raise FileNotFoundError("The file does not exist")
