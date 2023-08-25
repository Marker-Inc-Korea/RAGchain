import json
from abc import ABC, abstractmethod
from typing import List, Union, Any
from uuid import UUID

from KoPrivateGPT.DB import MongoDB, PickleDB
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.linker import RedisDBSingleton


class BaseRetrieval(ABC):
    def __init__(self):
        self.mongo_db = None
        self.pickle_db = None
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
    def split_ids(ids: List[UUID], db_origin_list):
        """
        mongo_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin[i].db_type == "mongo_db"]
        pickle_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin[i].db_type == "pickle_db"]
        """
        mongo_db_ids = []
        pickle_db_ids = []
        for i in range(len(db_origin_list)):
            if db_origin_list[i]['db_type'] == "mongo_db":
                mongo_db_ids.append(ids[i])
            elif db_origin_list[i]['db_type'] == "pickle_db":
                pickle_db_ids.append(ids[i])
            else:
                pass
        return mongo_db_ids, pickle_db_ids

    def fetch_data(self, ids: List[UUID]) -> List[Passage]:
        self.redis_db = RedisDBSingleton()
        db_origin_list = self.redis_db.get_json(ids)
        # Sometimes redis doesn't find the id, so we need to filter None.
        final_db_origin = [db_origin for db_origin in db_origin_list if db_origin is not None]
        mongo_db_ids, pickle_db_ids = self.split_ids(ids, final_db_origin)
        passage_list = []
        self.fetch_mongo_data(mongo_db_ids, final_db_origin, passage_list)
        self.fetch_pickle_data(pickle_db_ids, final_db_origin, passage_list)
        return passage_list

    def fetch_mongo_data(self, mongo_db_ids: List[UUID], db_origin_list, passage_list):
        if mongo_db_ids:
            # TODO: Because db_path is different, modify it to create only the minimum number of mongoDB objects.
            self.mongo_db = MongoDB(**db_origin_list[0]['db_path'])
            self.mongo_db.load()
            mongo_db_passage_list = self.mongo_db.fetch(mongo_db_ids)
            passage_list.append(mongo_db_passage_list)

    def fetch_pickle_data(self, pickle_db_ids: List[UUID], db_origin_list, passage_list):
        if pickle_db_ids:
            self.pickle_db = PickleDB(**db_origin_list[0]['db_path'])
            self.pickle_db.load()
            pickle_db_passage_list = self.pickle_db.fetch(pickle_db_ids)
            passage_list.append(pickle_db_passage_list)
