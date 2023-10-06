import os
from typing import Union
from uuid import UUID

import redis
from dotenv import load_dotenv

load_dotenv()


class RedisDBSingleton:
    __instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if self._is_initialized:
            return

        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
        db_name = os.getenv("REDIS_DB_NAME")
        password = os.getenv("REDIS_PW")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db_name,
            decode_responses=True,
            password=password
        )
        self._is_initialized = True

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
