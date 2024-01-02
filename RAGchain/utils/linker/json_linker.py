import json
import os
import warnings
from typing import Union, List
from uuid import UUID

from RAGchain.utils.linker.base import BaseLinker, NoIdWarning, NoDataWarning


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

    def put_json(self, ids: List[Union[UUID, str]], json_data_list: List[dict]):
        for i in range(len(ids)):
            self.data[str(ids[i])] = json_data_list[i]
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)

    def get_json(self, ids: List[Union[UUID, str]]):
        assert len(ids) > 0, "ids must be a non-empty list"
        str_ids = [str(find_id) for find_id in ids]
        results = []
        for _id in str_ids:
            if _id not in self.data:
                warnings.warn(f"ID {_id} not found in Linker", NoIdWarning)
                results.append(None)
            else:
                if self.data[_id] is None:
                    warnings.warn(f"Data {_id} not found in Linker", NoDataWarning)
                results.append(self.data[_id])
        return results

    def flush_db(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)
        else:
            raise FileNotFoundError("The file does not exist")

    def delete_json(self, ids: List[Union[UUID, str]]):
        for _id in ids:
            del self.data[str(_id)]
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)
