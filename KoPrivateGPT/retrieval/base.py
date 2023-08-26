import json
from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict
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
    def split_db_type(ids: List[UUID], db_origin_list):
        mongo_db_ids = []
        mongo_db_path = []
        pickle_db_ids = []
        pickle_db_path = []
        for i in range(len(db_origin_list)):
            if db_origin_list[i]['db_type'] == "mongo_db":
                mongo_db_ids.append(ids[i])
                mongo_db_path.append(db_origin_list[i]['db_path'])
            elif db_origin_list[i]['db_type'] == "pickle_db":
                pickle_db_ids.append(ids[i])
                pickle_db_path.append(db_origin_list[i]['db_path'])
            else:
                raise ValueError(f"Unknown db type: {db_origin_list[i]['db_type']}")
        return mongo_db_ids, mongo_db_path, pickle_db_ids, pickle_db_path

    def fetch_data(self, ids: List[UUID]) -> List[Passage]:
        self.redis_db = RedisDBSingleton()
        db_origin_list = self.redis_db.get_json(ids)
        # Sometimes redis doesn't find the id, so we need to filter that db_origin is None.
        final_db_origin = [db_origin for db_origin in db_origin_list if db_origin is not None]
        mongo_db_ids, mongo_db_path, pickle_db_ids, pickle_db_path = self.split_db_type(ids, final_db_origin)
        passage_list = self.fetch_each_db(mongo_db_ids, pickle_db_ids, mongo_db_path, pickle_db_path)
        return passage_list

    def fetch_each_db(self, mongo_db_ids, pickle_db_ids, mongo_db_path, pickle_db_path) -> List[Passage]:
        if not pickle_db_ids:
            passage_list = self.fetch_mongo_data(mongo_db_ids, mongo_db_path)
        elif not mongo_db_ids:
            passage_list = self.fetch_pickle_data(pickle_db_ids, pickle_db_path)
        else:
            passage_list = self.fetch_mongo_data(mongo_db_ids, mongo_db_path) + self.fetch_pickle_data(pickle_db_ids,
                                                                                                       pickle_db_path)
        return passage_list

    def fetch_mongo_data(self, mongo_db_ids: List[UUID], mongo_db_path) -> List[Passage]:
        check_dict = self.duplicate_path_check(mongo_db_path)
        fetch_list = []
        for db_path, index_list in check_dict.items():
            real_mongo_db_ids = []
            for i in index_list:
                real_mongo_db_ids = [mongo_db_ids[i]]
            self.mongo_db = MongoDB(**dict(db_path))
            self.mongo_db.load()
            fetch_list.append(self.mongo_db.fetch(real_mongo_db_ids))
        passage_list = self.flatten_list(fetch_list)
        return passage_list

    def fetch_pickle_data(self, pickle_db_ids: List[UUID], pickle_db_path) -> List[Passage]:
        check_dict = self.duplicate_path_check(pickle_db_path)

        self.pickle_db = PickleDB(**pickle_db_path[0])
        self.pickle_db.load()
        return self.pickle_db.fetch(pickle_db_ids)

    @staticmethod
    def duplicate_path_check(db_path: List) -> dict[tuple, list[Any]]:
        """
        mongo_db_path = [{'mongo_url': '~', 'db_name': '~', 'collection_name': '~'}, {}, ...]
        for example,
        result = {(('mongo_url': '~'), ('db_name': '~'), ('collection_name': '~')): [0, 1, 2], {}: [3]}
        """
        check_duplicate = []
        result = {}
        for index, item in enumerate(db_path):
            if item in check_duplicate:
                result[tuple(item.items())].append(index)
            else:
                check_duplicate.append(item)
                result[tuple(item.items())] = [index]
        return result

    @staticmethod
    def flatten_list(nested_list):
        final_list = []
        for item in nested_list:
            final_list.extend(item)
        return final_list
