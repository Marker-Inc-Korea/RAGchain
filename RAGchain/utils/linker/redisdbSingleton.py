import os
import warnings
from typing import Union
from uuid import UUID

import redis
from dotenv import load_dotenv
from RAGchain.utils.linker.base import BaseLinker

load_dotenv()


class RedisDBSingleton(BaseLinker):
    """
    RedisDBSingleton is a singleton class that manages redis.
    We use redis to link DB and passage ids that stores in retrievals.
    """

    def __init__(self):
        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
        db_name = os.getenv("REDIS_DB_NAME")
        password = os.getenv("REDIS_PW")

        if host is None:
            raise ValueError("Please set REDIS_HOST to environment variable")
        if port is None:
            raise ValueError("Please set REDIS_PORT to environment variable")
        if db_name is None:
            raise ValueError("Please set REDIS_DB_NAME to environment variable")
        if password is None:
            warnings.warn("REDIS_PW is not set. You can set REDIS_PW to environment variable", UserWarning)

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db_name,
            decode_responses=True,
            password=password
        )

    def get_json(self, ids: list[Union[UUID, str]]):
        # redis only accept str type key
        str_ids = [str(find_id) for find_id in ids]
        return [self.client.json().get(find_id) for find_id in str_ids]

    def connection_check(self):
        return self.client.ping()

    def flush_db(self):
        self.client.flushdb()

    def __del__(self):
        self.client.close()

    def put_json(self, id: Union[UUID, str], json_data: dict):
        self.client.json().set(str(id), '$', json_data)
