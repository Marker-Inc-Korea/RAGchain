from typing import List, Optional, Union
from uuid import UUID

import pymongo

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
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
            db_origin_dict = db_origin.to_dict()
            self.redis_db.client.json().set(str(passage.id), '$', db_origin_dict)

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        dict_passages = list(self.collection.find({"_id": {"$in": ids}}))
        return [Passage(id=dict_passage['_id'], **dict_passage) for dict_passage in dict_passages]

    def search(self,
               id: Optional[List[Union[UUID, str]]] = None,
               content: Optional[List[str]] = None,
               filepath: Optional[List[str]] = None,
               **kwargs
               ) -> List[Passage]:
        """
        With this function, you can use mongoDB's find function.
        :params filter_dict: dict, query dict for mongoDB's find function.
        :return: List[Passage], list of Passage extract from the result of filter_dict query to MongoDB.
        """
        filter_dict = {}
        if id is not None:
            filter_dict["_id"] = {'$in': id}
        if content is not None:
            filter_dict["content"] = {'$in': content}
        if filepath is not None:
            filter_dict["filepath"] = {'$in': filepath}
        if kwargs is not None and len(kwargs) > 0:
            for key, value in kwargs.items():
                filter_dict[f'metadata_etc.{key}'] = {'$in': value}

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
