import itertools
from abc import ABC, abstractmethod
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.DB import MongoDB, PickleDB
from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage, DBOrigin
from KoPrivateGPT.utils.linker import RedisDBSingleton


class BaseRetrieval(ABC):
    def __init__(self):
        self.db_instance_list: List[BaseDB] = []
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

    def fetch_data(self, ids: List[Union[UUID, str]]) -> List[Passage]:
        db_origin_list = self.redis_db.get_json(ids)
        # Sometimes redis doesn't find the id, so we need to filter that db_origin is None.
        filter_db_origin = list(filter(lambda db_origin: db_origin is not None, db_origin_list))
        # Check duplicated db origin in one retrieval.
        final_db_origin = self.duplicate_check(filter_db_origin)
        # fetch data from each db
        passage_list = self.fetch_each_db(final_db_origin, ids)
        return passage_list

    def fetch_each_db(self, final_db_origin: dict[tuple, list[int]], ids: List[Union[UUID, str]]):
        """
        check_dict = {(("db_type": "mongo_db"),
            (('mongo_url': "~"), ('db_name': "~"), ('collection_name': "~"))): [0,  2], ...}
        """
        fetch_list = []
        for item in final_db_origin.items():
            # make tuple to dict
            # item[0] = db_origin:tuple, item[1] = index:list
            db_origin = dict(item[0])
            dict_db_path = dict(db_origin['db_path'])
            # make db instance
            db = self.is_created(db_origin['db_type'], dict_db_path)
            db.load()
            # make each id list
            each_ids = [ids[i] for i in item[1]]
            # fetch data
            fetch_data = (db.fetch(each_ids))
            fetch_list.append(fetch_data)
        # make flatten list(passage_list) from fetch_list
        passage_list = list(itertools.chain.from_iterable(fetch_list))
        return passage_list

    def is_created(self, db_type: str, db_path: dict):
        if not self.db_instance_list:
            db = self.create_db(db_type, db_path)
            self.db_instance_list.append(db)
            return db
        else:
            db_origin_list = [instance.get_db_origin() for instance in self.db_instance_list]
            db_origin = DBOrigin(db_type=db_type, db_path=db_path)
            if db_origin in db_origin_list:
                return self.db_instance_list[db_origin_list.index(db_origin)]
            else:
                db = self.create_db(db_type, db_path)
                self.db_instance_list.append(db)
                return db

    @staticmethod
    def create_db(db_type: str, db_path: dict) -> BaseDB:
        """
        selector-ModuleSelector cant import because of circular import.
        """
        if db_type == "mongo_db":
            return MongoDB(**db_path)
        elif db_type == "pickle_db":
            return PickleDB(**db_path)
        else:
            raise ValueError(f"Unknown db type: {db_type}")

    @staticmethod
    def duplicate_check(db_origin_list: list[dict]) -> dict[tuple, list[int]]:
        """
        Check duplicated db origin in one retrieval.
        For example,
        db_origin = {"db_type": "mongo_db",
            "db_path": {"mongo_url": "...", "db_name": "...", "collection_name": "..."}}
        result = {(("db_type": "mongo_db"),
            ('db_path',(('mongo_url': "..."), ('db_name': "..."), ('collection_name': "...")))): [0,  2], ...}
        """
        check_origin_duplicate = []
        result = {}
        for index, db_origin in enumerate(db_origin_list):
            # db_origin(dict) to tuple
            tuple_db_origin = tuple(db_origin.items())
            # replace db_path(dict) to tuple
            tuple_final = tuple([(key, tuple(value.items())) if key == "db_path" else (key, value)
                                 for key, value in tuple_db_origin])
            # check duplicated db instance with equal method
            if db_origin in check_origin_duplicate:
                result[tuple_final].append(index)
            else:
                check_origin_duplicate.append(db_origin)
                result[tuple_final] = [index]
        return result
