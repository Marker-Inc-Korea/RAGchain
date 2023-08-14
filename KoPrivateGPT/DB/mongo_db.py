from typing import List, Any

import pymongo
from bson import Binary, UuidRepresentation

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
from uuid import UUID


class MongoDB(BaseDB):
    def __init__(self, mongo_url: str, db_name: str, collection_name: str, *args, **kwargs):
        self.client = None
        self.db = None
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None

    @property
    def db_type(self) -> str:
        return 'MongoDB'

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
            passage_to_dict = passage.to_dict()
            self.collection.insert_one(passage_to_dict)

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        passage_list = []
        for find_id in ids:
            dict_passage = self.collection.find_one({"_id": find_id})
            passage = Passage(**dict_passage)
            passage_list.append(passage)
        return passage_list

    def search(self, filter: Any) -> List[Passage]:
        raise NotImplementedError("MongoDB does not support search method")

    def set_db(self):
        self.client = pymongo.MongoClient(self.mongo_url, uuidRepresentation='standard')
        if self.db_name not in self.client.list_database_names():
            raise ValueError(f'{self.db_name} does not exists')
        self.db = self.client.get_database(self.db_name)
