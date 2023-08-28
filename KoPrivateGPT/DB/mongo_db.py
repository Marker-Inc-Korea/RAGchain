from typing import List, Any, Dict

import pymongo

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
from uuid import UUID

from KoPrivateGPT.schema.db_origin import DBOrigin
from KoPrivateGPT.utils.linker.redisdbSingleton import RedisDBSingleton


class MongoDB(BaseDB):
    def __init__(self, mongo_url: str, db_name: str, collection_name: str, *args, **kwargs):
        self.client = None
        self.db = None
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        self.redis_db = RedisDBSingleton()

    @property
    def db_type(self) -> str:
        return 'mongo_db'

    def create(self):
        self.set_db()
        if self.collection_name in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} already exists')
        self.collection = self.db.create_collection(self.collection_name)

    def load(self):
        self.set_db()
        if self.collection_name not in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} does not exist')
        self.collection = self.db.get_collection(self.collection_name)

    def create_or_load(self):
        self.set_db()
        if self.collection_name in self.db.list_collection_names():
            self.load()
        else:
            self.create()

    def save(self, passages: List[Passage]):
        for passage in passages:
            # save to mongoDB
            passage_to_dict = passage.to_dict()
            self.collection.insert_one(passage_to_dict)
            # save to redisDB
            db_origin = self.get_db_origin()
            db_origin_json = db_origin.to_json()
            self.redis_db.client.json().set(str(passage.id), '$', db_origin_json)

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        passage_list = []
        for find_id in ids:
            dict_passage = self.collection.find_one({"_id": find_id})
            if dict_passage is None:
                raise ValueError(f'{find_id} This _id does not exist in {self.collection_name} collection')
            passage = Passage(id=dict_passage['_id'], **dict_passage)
            passage_list.append(passage)
        return passage_list

    def search(self, filter_dict: Dict[str, Any]) -> List[Passage]:
        """
        With this function, you can use mongoDB's find function.
        :params filter_dict: dict, query dict for mongoDB's find function.
        :return: List[Passage], list of Passage extract from the result of filter_dict query to MongoDB.
        """
        cursor = self.collection.find(filter_dict)
        return [Passage(id=passage['_id'], **passage) for passage in cursor]

    def set_db(self):
        self.client = pymongo.MongoClient(self.mongo_url, uuidRepresentation='standard')
        if self.db_name not in self.client.list_database_names():
            raise ValueError(f'{self.db_name} does not exists')
        self.db = self.client.get_database(self.db_name)

    def get_db_origin(self) -> DBOrigin:
        db_path = {'mongo_url': self.mongo_url, 'db_name': self.db_name, 'collection_name': self.collection_name}
        return DBOrigin(db_type=self.db_type, db_path=db_path)
