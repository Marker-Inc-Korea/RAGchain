import os
from dotenv import load_dotenv

import pymongo

from RAGchain.utils.linker.base import BaseLinker

load_dotenv()


class MongoLinker(BaseLinker):
    """
    MongoLinker is a singleton class that manages MongoDB.
    """

    def __init__(self):
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        collection_name = os.getenv("MONGO_COLLECTION_NAME")

        if mongo_url is None:
            raise ValueError("Please set MONGO_URL to environment variable")
        if db_name is None:
            raise ValueError("Please set MONGO_DB_NAME to environment variable")
        if collection_name is None:
            raise ValueError("Please set MONGO_COLLECTION_NAME to environment variable")

        self.client = pymongo.MongoClient(mongo_url)
        self.db = None
        self.collection = None
        self.db_name = db_name
        self.collection_name = collection_name
        self.create_or_load_collection()

    def create_collection(self):
        """
        Create a collection in MongoDB that can be used to store DB origin.
        """
        self.db = self.client[self.db_name]
        if self.collection_name in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} already exists')
        self.collection = self.db.create_collection(self.collection_name)

    def load_collection(self):
        """
        Load a collection in MongoDB that can be used to store DB origin.
        """
        self.db = self.client[self.db_name]
        if self.collection_name not in self.db.list_collection_names():
            raise ValueError(f'{self.collection_name} does not exist')
        self.collection = self.db.get_collection(self.collection_name)

    def create_or_load_collection(self):
        """
        Create a collection if it does not exist, otherwise load it.
        """
        self.db = self.client[self.db_name]
        if self.collection_name in self.db.list_collection_names():
            self.load_collection()
        else:
            self.create_collection()

    def get_json(self, ids):
        return [self.collection.find_one({'_id': id}) for id in ids]

    def put_json(self, id, json_data):
        self.collection.insert_one({'_id': id, 'data': json_data})

    def flush_db(self):
        self.client.drop_database(self.db_name)
