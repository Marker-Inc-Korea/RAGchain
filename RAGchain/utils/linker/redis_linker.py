import os
import warnings
from typing import Union, List
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

    def get_json(self, ids: List[Union[UUID, str]]):
        assert len(ids) > 0, "ids must be a non-empty list"
        # redis only accept str type key
        str_ids = [str(find_id) for find_id in ids]

        response = self.client.json().mget(str_ids, '$')
        results = []
        for i, sublist in enumerate(response):
            if sublist is None:
                warnings.warn(f"ID {str_ids[i]} not found in Linker", NoIdWarning)
                results.append(None)
            else:
                results.append(sublist[0])
                if sublist[0] is None:
                    warnings.warn(f"Data {str_ids[i]} not found in Linker", NoDataWarning)
        return results

    def connection_check(self):
        return self.client.ping()

    def flush_db(self):
        self.client.flushdb()

    def __del__(self):
        self.client.close()

    def put_json(self, ids: List[Union[UUID, str]], json_data_list: List[dict]):
        str_ids = [str(find_id) for find_id in ids]
        triplets = []
        for i in range(len(str_ids)):
            triplets.append((str_ids[i], '$', json_data_list[i]))
        self.client.json().mset(triplets)

    def delete_json(self, ids: List[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        self.client.delete(*str_ids)
