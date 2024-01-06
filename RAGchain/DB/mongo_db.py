from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

import pymongo
from pymongo import UpdateOne

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

    def save(self, passages: List[Passage], upsert: bool = False):
        """Saves the passages to MongoDB collection."""
        # Setting up files for saving to 'mongodb'
        dict_passages = list(map(lambda x: x.to_dict(), passages))
        # Setting up files for saving to 'linker'
        id_list = list(map(lambda x: str(x.id), passages))
        db_origin_list = [self.get_db_origin().to_dict() for _ in passages]

        # save to 'mongodb'
        if upsert:
            db_id_list = [doc['_id'] for doc in self.collection.find({'_id': {'$in': id_list}}, {'_id': 1})]
            # Create a dictionary of passages with id as key
            dict_passages_dict = {_id: dict_passages[i] for i, _id in enumerate(id_list)}
            if len(db_id_list) > 0:
                requests = [UpdateOne({'_id': _id},
                                  {'$set': dict_passages_dict[_id]}, upsert=True) for _id in db_id_list]
                self.collection.bulk_write(requests)
            not_duplicated_ids = [id for id in id_list if id not in db_id_list]
            not_duplicated_passages = [dict_passages_dict[_id] for _id in not_duplicated_ids]
            if len(not_duplicated_passages) > 0:
                self.collection.insert_many(not_duplicated_passages)
        else:
            self.collection.insert_many(dict_passages)

        # save to 'linker'
        linker.put_json(id_list, db_origin_list)

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
               importance: Optional[List[int]] = None,
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
        if importance is not None:
            filter_dict["importance"] = {'$in': importance}
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
