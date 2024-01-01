import os
import warnings
from typing import Union
from uuid import UUID

import redis
from dotenv import load_dotenv
from RAGchain.utils.linker.base import BaseLinker, NoIdWarning, NoDataWarning

load_dotenv()


class RedisLinker(BaseLinker):
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
        no_id_indices = []
        for i, find_id in enumerate(str_ids):
            if find_id not in self.client.keys() or self.client.keys() is None:
                warnings.warn(f"ID {find_id} not found in Linker", NoIdWarning)
                no_id_indices.append(i)
                str_ids.pop(i)
        # if all ids are not found in redis, return None because if str_ids is empty, mget will raise error.
        if len(str_ids) == 0:
            return [None]
        results = self.client.json().mget(str_ids, '$')
        flattened_data = [item for sublist in results for item in sublist]
        for i, data in enumerate(flattened_data):
            if data is None:
                warnings.warn(f"Data {str_ids[i]} not found in Linker", NoDataWarning)
        for index in no_id_indices:
            flattened_data.insert(index, None)
        return flattened_data

    def connection_check(self):
        return self.client.ping()

    def flush_db(self):
        self.client.flushdb()

    def __del__(self):
        self.client.close()

    def put_json(self, id: Union[UUID, str], json_data: dict):
        self.client.json().set(str(id), '$', json_data)

    def delete_json(self, id: Union[UUID, str]):
        self.client.delete(str(id))
