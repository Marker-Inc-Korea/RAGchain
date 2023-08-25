import json
from abc import ABC, abstractmethod
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.pipeline.selector import ModuleSelector
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.linker import RedisDBSingleton


class BaseRetrieval(ABC):
    def __init__(self):
        self.redis_db = RedisDBSingleton()

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        pass

    @abstractmethod
    def ingest(self, passages: List[Passage]):
        pass

    @abstractmethod
    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        pass

    # If you want more db type, you can add more db type below this line.
    @staticmethod
    def split_ids(ids: List[UUID], db_origin_list: List[json]):
        mongo_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin.db_type == "mongo_db"]
        pickle_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin.db_type == "pickle_db"]
        return mongo_db_ids, pickle_db_ids

    def fetch_data(self, ids: List[UUID]) -> List[Passage]:
        self.redis_db = RedisDBSingleton()
        db_origin_list = self.redis_db.get_json(ids)
        mongo_db_ids, pickle_db_ids = self.split_ids(ids, db_origin_list)
        passage_list = []
        self.fetch_mongo_data(mongo_db_ids, db_origin_list, passage_list)
        self.fetch_pickle_data(pickle_db_ids, db_origin_list, passage_list)
        return passage_list

    @staticmethod
    def fetch_mongo_data(mongo_db_ids: List[UUID], db_origin_list, passage_list):
        if mongo_db_ids:
            mongo_db = ModuleSelector("db").select("mongo_db").get(**db_origin_list[0].db_path)
            mongo_db.load()
            mongo_db_passage_list = mongo_db.fetch(mongo_db_ids)
            passage_list.append(mongo_db_passage_list)

    @staticmethod
    def fetch_pickle_data(pickle_db_ids: List[UUID], db_origin_list, passage_list):
        if pickle_db_ids:
            pickle_db = ModuleSelector("db").select("pickle_db").get(**db_origin_list[0].db_path)
            pickle_db.load()
            pickle_db_passage_list = pickle_db.fetch(pickle_db_ids)
            passage_list.append(pickle_db_passage_list)
