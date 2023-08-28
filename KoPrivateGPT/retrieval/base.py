from abc import ABC, abstractmethod
from typing import List, Union, Dict
from uuid import UUID

from KoPrivateGPT.DB import MongoDB, PickleDB
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.linker import RedisDBSingleton


class BaseRetrieval(ABC):
    def __init__(self):
        self.db = None
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

    def fetch_data(self, ids: List[UUID]) -> List[Passage]:
        db_origin_list = self.redis_db.get_json(ids)
        # Sometimes redis doesn't find the id, so we need to filter that db_origin is None.
        filter_db_origin = self.none_filter(db_origin_list)
        # Check duplicated db instance
        final_db_origin = self.duplicate_check(filter_db_origin)
        # fetch data from each db
        passage_list = self.fetch_each_db(final_db_origin, ids)
        return passage_list

    def fetch_each_db(self, final_db_origin: dict[tuple, list[int]], ids: List[UUID]):
        """
        check_dict = {(("db_type": "mongo_db"),
            (('mongo_url': "~"), ('db_name': "~"), ('collection_name': "~"))): [0,  2], ...}
        """
        fetch_list = []
        for item in final_db_origin.items():
            # make tuple to dict
            # item[0] = (db_origin:tuple), item[1] = (index:list)
            db_origin = dict(item[0])
            dict_db_path = dict(db_origin['db_path'])
            # make db instance
            self.create_db_instance(db_origin['db_type'], dict_db_path)
            self.db.load()
            # make each id list
            each_ids = [ids[i] for i in item[1]]
            # fetch data
            fetch_data = (self.db.fetch(each_ids))
            fetch_list.append(fetch_data)
        passage_list = self.flatten_list(fetch_list)
        return passage_list

    def create_db_instance(self, db_type: str, db_path: dict):
        """
        selector-ModuleSelector cant import because of circular import.
        """
        if db_type == "mongo_db":
            self.db = MongoDB(**db_path)
        elif db_type == "pickle_db":
            self.db = PickleDB(**db_path)
        else:
            raise ValueError(f"Unknown db type: {db_type}")

    @staticmethod
    def duplicate_check(db_origin_list: list[dict]) -> dict[tuple, list[int]]:
        """
        For example,
        db_origin = {"db_type": "mongo_db",
            "db_path": {"mongo_url": "...", "db_name": "...", "collection_name": "..."}}
        check_dict = {(("db_type": "mongo_db"),
            ('db_path',(('mongo_url': "..."), ('db_name': "..."), ('collection_name': "...")))): [0,  2], ...}
        """
        check_duplicate = []
        result = {}
        for index, db_origin in enumerate(db_origin_list):
            # db_origin(dict) to tuple
            tuple_db_origin = tuple(db_origin.items())
            # replace db_path(dict) to tuple
            tuple_final = tuple([(key, tuple(value.items())) if key == "db_path" else (key, value)
                                 for key, value in tuple_db_origin])
            if db_origin in check_duplicate:
                result[tuple_final].append(index)
            else:
                check_duplicate.append(db_origin)
                result[tuple_final] = [index]
        return result

    @staticmethod
    def flatten_list(nested_list):
        final_list = []
        for item in nested_list:
            final_list.extend(item)
        return final_list

    @staticmethod
    def none_filter(db_origin_list: List[Dict]) -> List[Dict]:
        final_db_origin = []
        for db_origin in db_origin_list:
            if db_origin is None:
                raise ValueError("Redis doesn't find the id")
            else:
                final_db_origin.append(db_origin)
        return final_db_origin
