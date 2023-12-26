from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

import pymongo

from RAGchain import linker
from RAGchain.DB.base import BaseDB
from RAGchain.schema import Passage
from RAGchain.schema.db_origin import DBOrigin


class MongoDB(BaseDB):
    """
    MongoDB class for using MongoDB as a database for passage contents.
    """
    def __init__(self, mongo_url: str, db_name: str, collection_name: str, *args, **kwargs):
        """
        :param mongo_url: str, the url of mongoDB server.
        :param db_name: str, the name of mongoDB database.
        :param collection_name: str, the name of collection in mongoDB database.
        """
        self.client = None
        self.db = None
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None

    @property
    def db_type(self) -> str:
        """Returns the type of the database as a string."""
        return 'mongo_db'

    def create(self):
        """Creates the collection in the MongoDB database. Raises a `ValueError` if the collection already exists."""
        self.set_db()
        if self.collection_name in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} already exists')
        self.collection = self.db.create_collection(self.collection_name)

    def load(self):
        """Loads the collection from the MongoDB database. Raises a `ValueError` if the collection does not exist."""
        self.set_db()
        if self.collection_name not in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} does not exist')
        self.collection = self.db.get_collection(self.collection_name)

    def create_or_load(self):
        """Creates the collection if it does not exist, otherwise loads it."""
        self.set_db()
        if self.collection_name in self.db.list_collection_names():
            self.load()
        else:
            self.create()

    def save(self, passages: List[Passage]):
        """Saves the passages to MongoDB collection."""
        for passage in passages:
            # save to mongoDB
            passage_to_dict = passage.to_dict()
            self.collection.insert_one(passage_to_dict)
            # save to redisDB
            db_origin = self.get_db_origin()
            db_origin_dict = db_origin.to_dict()
            linker.put_json(str(passage.id), db_origin_dict)

    def fetch(self, ids: List[UUID]) -> List[Passage]:
        """Fetches the passages from MongoDB collection by their passage ids."""
        dict_passages = list(self.collection.find({"_id": {"$in": ids}}))
        result = list()
        for dict_passage in dict_passages:
            _id = dict_passage.pop('_id')
            result.append(Passage(id=_id, **dict_passage))
        return result

    def search(self,
               id: Optional[List[Union[UUID, str]]] = None,
               content: Optional[List[str]] = None,
               filepath: Optional[List[str]] = None,
               content_datetime_range: Optional[List[tuple[datetime, datetime]]] = None,
               **kwargs
               ) -> List[Passage]:
        filter_dict = {}
        if id is not None:
            filter_dict["_id"] = {'$in': id}
        if content is not None:
            filter_dict["content"] = {'$in': content}
        if filepath is not None:
            filter_dict["filepath"] = {'$in': filepath}
        if content_datetime_range is not None:
            filter_dict["$or"] = [{'content_datetime': {'$gte': start, '$lte': end}} for start, end in
                                  content_datetime_range]
        if kwargs is not None and len(kwargs) > 0:
            for key, value in kwargs.items():
                filter_dict[f'metadata_etc.{key}'] = {'$in': value}

        cursor = self.collection.find(filter_dict)
        result = list()
        for passage in cursor:
            _id = passage.pop('_id')
            result.append(Passage(id=_id, **passage))
        return result

    def set_db(self):
        self.client = pymongo.MongoClient(self.mongo_url, uuidRepresentation='standard')
        if self.db_name not in self.client.list_database_names():
            raise ValueError(f'{self.db_name} does not exists')
        self.db = self.client.get_database(self.db_name)

    def get_db_origin(self) -> DBOrigin:
        """
        Returns the DBOrigin object representing the MongoDB database.
        """
        db_path = {'mongo_url': self.mongo_url, 'db_name': self.db_name, 'collection_name': self.collection_name}
        return DBOrigin(db_type=self.db_type, db_path=db_path)
