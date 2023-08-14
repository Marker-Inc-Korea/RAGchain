from typing import List, Any

import pymongo
from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
from uuid import UUID


class MongoDB(BaseDB):
    def __init__(self, mongo_url: str, db_name: str):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = pymongo.MongoClient(self.mongo_url)
        self.db = self.client.get_database(self.db_name)
        self.collection = None

    @property
    def db_type(self) -> str:
        return 'MongoDB'

    def create(self, collection_name: str):
        if collection_name in self.db.list_collection_names():
            raise ValueError(f'{collection_name} already exists')
        if collection_name not in self.db.list_collection_names():
            self.collection = self.db.create_collection(collection_name)

    def load(self, collection_name: str):
        if collection_name not in self.db.list_collection_names():
            raise ValueError(f'{collection_name} does not exist')
        self.collection = self.db.get_collection(collection_name)

    def create_or_load(self, collection_name: str):
        if collection_name in self.db.list_collection_names():
            self.load(collection_name)
        else:
            self.create(collection_name)

    def save(self, passages: List[Passage]):
        for passage in passages:
            self.collection.insert_one(passage.to_json())

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        result = list(self.collection.find({"id": {"$in": ids}}))
        return result

    def search(self, filter: Any) -> List[Passage]:
        raise NotImplementedError("MongoDB does not support search method")
