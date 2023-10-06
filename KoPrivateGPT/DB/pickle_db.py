import os
import pickle
from typing import List, Optional, Union
from uuid import UUID

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.schema.db_origin import DBOrigin
from KoPrivateGPT.utils.linker import RedisDBSingleton
from KoPrivateGPT.utils.util import FileChecker


class PickleDB(BaseDB):

    def __init__(self, save_path: str, *args, **kwargs):
        FileChecker(save_path).check_type(file_types=['.pickle', '.pkl'])
        self.save_path = save_path
        self.db: List[Passage] = list()
        self.redis_db = RedisDBSingleton()

    @property
    def db_type(self) -> str:
        return 'pickle_db'

    def create(self):
        if os.path.exists(self.save_path):
            raise FileExistsError(f'{self.save_path} already exists')
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        self.save_path = self.save_path

    def load(self):
        if not FileChecker(self.save_path).check_type(file_types=['.pickle', '.pkl']).is_exist():
            raise FileNotFoundError(f'{self.save_path} does not exist')
        with open(self.save_path, 'rb') as f:
            self.db = pickle.load(f)

    def create_or_load(self):
        if os.path.exists(self.save_path):
            self.load()
        else:
            self.create()

    def save(self, passages: List[Passage]):
        # save to pickleDB
        self.db.extend(passages)
        self._write_pickle()
        # save to redisDB
        db_origin = self.get_db_origin()
        db_origin_dict = db_origin.to_dict()
        [self.redis_db.client.json().set(str(passage.id), '$', db_origin_dict) for passage in passages]

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        result = list(filter(lambda x: x.id in ids, self.db))
        return result

    def search(self,
               id: Optional[List[Union[UUID, str]]] = None,
               content: Optional[List[str]] = None,
               filepath: Optional[List[str]] = None,
               **kwargs) -> List[Passage]:
        """
        This function is an implicit AND operation,
        which is return Passage that matches all values to corresponding keys in filter_dict.
        When the match value is not exist, return empty list.
        """

        def is_default_elem(filter_key: str) -> bool:
            return filter_key in ['id', 'content', 'filepath']

        filter_dict = dict()
        if id is not None:
            filter_dict['id'] = id
        if content is not None:
            filter_dict['content'] = content
        if filepath is not None:
            filter_dict['filepath'] = filepath
        if kwargs is not None and len(kwargs) > 0:
            for key, value in kwargs.items():
                filter_dict[key] = value

        result = list(
            filter(
                lambda x: all(
                    getattr(x, key) in value if is_default_elem(key) else x.metadata_etc.get(key) in value
                    for key, value in filter_dict.items()
                ),
                self.db
            )
        )
        return result

    def _write_pickle(self):
        with open(self.save_path, 'wb') as w:
            pickle.dump(self.db, w)

    def get_db_origin(self) -> DBOrigin:
        return DBOrigin(db_type=self.db_type, db_path={'save_path': self.save_path})
