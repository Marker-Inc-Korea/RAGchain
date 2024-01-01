from RAGchain.utils.linker.base import BaseLinker, NoIdWarning, NoDataWarning
from typing import Union
from uuid import UUID

import os
import json
import warnings


class JsonLinker(BaseLinker):
    """
    JsonLinker is a singleton class that allows the role of a linker
    to be played locally to use JSON file without using an external DB like redis or dynamo.
    """
    def __init__(self):
        json_path = os.getenv("JSON_LINKER_PATH")

        if json_path is None:
            raise ValueError("Please set JSON_LINKER_PATH to environment variable")

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
        self.data[str(id)] = json_data
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)

    def get_json(self, ids: list[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        data_list = []
        for find_id in str_ids:
            # Check if id exists in json linker
            if find_id not in self.data:
                warnings.warn(f"ID {find_id} not found in JsonLinker", NoIdWarning)
                continue
            else:
                data = self.data.get(find_id)
                # Check if data exists in json linker
                if data is None:
                    warnings.warn(f"Data {find_id} not found in JsonLinker", NoDataWarning)
                    continue
                else:
                    data_list.append(data)
        return data_list

    def flush_db(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)
        else:
            raise FileNotFoundError("The file does not exist")
