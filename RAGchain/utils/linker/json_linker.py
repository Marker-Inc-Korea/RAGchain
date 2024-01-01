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

    def put_json(self, ids: list[Union[UUID, str]], json_data_list: list[dict]):
        for i in range(len(ids)):
            self.data[str(ids[i])] = json_data_list[i]
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)

    def get_json(self, ids: list[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        no_id_indices = []
        for i, find_id in enumerate(str_ids):
            if find_id not in self.data.keys():
                warnings.warn(f"ID {find_id} not found in Linker", NoIdWarning)
                no_id_indices.append(i)
                str_ids.pop(i)
        # if all ids are not found in redis, return None because if str_ids is empty, mget will raise error.
        if len(str_ids) == 0:
            return [None]
        results = [self.data.get(str_id) for str_id in str_ids]
        for i, data in enumerate(results):
            if data is None:
                warnings.warn(f"Data {str_ids[i]} not found in Linker", NoDataWarning)
        for index in no_id_indices:
            results.insert(index, None)
        return results

    def flush_db(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)
        else:
            raise FileNotFoundError("The file does not exist")

    def delete_json(self, id: Union[UUID, str]):
        del self.data[str(id)]
        with open(self.json_path, "w") as f:
            json.dump(self.data, f)
