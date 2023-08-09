from typing import Any, List, Dict
from uuid import UUID
from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
import os
import pickle


class PickleDB(BaseDB):
    def __init__(self, save_path):
        self.save_path = save_path
        self.db = list()

    @property
    def db_type(self) -> str:
        return 'local'

    def create(self):
        if os.path.exists(self.save_path):
            raise FileExistsError(f'{self.save_path} already exists')
        self.save_path = self.save_path

    def load(self, *args, **kwargs):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f'{self.save_path} does not exist')
        self.db = self._load_pickle()

    def create_or_load(self, *args, **kwargs):
        if os.path.exists(self.save_path):
            self.load()
        else:
            self.create()

    def save(self, passages: List[Passage]):
        self.db.extend(passages)
        self._write_pickle()

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        result = list(filter(lambda x: x.id in ids, self.db))
        return result

    def search(self, filter: Dict[str, str]) -> List[Passage]:
        raise NotImplementedError("PickleDB does not support search method")

    def _load_pickle(self) -> Any:
        assert (os.path.splitext(self.save_path)[-1] == '.pickle')
        with open(self.save_path, 'rb') as f:
            return pickle.load(f)

    def _write_pickle(self):
        with open(self.save_path, 'wb') as w:
            pickle.dump(self.db, w)
