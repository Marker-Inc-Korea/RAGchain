from typing import List, Any

import pymongo
from bson import Binary, UuidRepresentation

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage
from uuid import UUID


class MongoDB(BaseDB):
    def __init__(self, mongo_url: str, db_name: str, collection_name: str, *args, **kwargs):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = pymongo.MongoClient(self.mongo_url, uuidRepresentation='standard')
        self.db = self.client.get_database(self.db_name)
        self.collection_name = collection_name
        self.collection = None

    @property
    def db_type(self) -> str:
        return 'MongoDB'

    def create(self):
        if self.collection_name in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} already exists')
        if self.collection_name not in self.db.list_collection_names():
            self.collection = self.db.create_collection(self.collection_name)

    def load(self):
        if self.collection_name not in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} does not exist')
        self.collection = self.db.get_collection(self.collection_name)

    def create_or_load(self):
        if self.collection_name in self.db.list_collection_names():
            self.load()
        else:
            self.create()

    def save(self, passages: List[Passage]):
        for passage in passages:
            passage_to_dict = {
                "_id": passage.id, "content": passage.content, "filepath": passage.filepath,
                "previous_passage_id": passage.previous_passage_id, "next_passage_id": passage.next_passage_id,
                "metadata_etc": passage.metadata_etc
            }
            self.collection.insert_one(passage_to_dict)

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        result = list(self.collection.find({"id": {"$in": ids}}))
        return result

    def search(self, filter: Any) -> List[Passage]:
        raise NotImplementedError("MongoDB does not support search method")
