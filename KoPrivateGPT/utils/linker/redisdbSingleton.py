from uuid import UUID
import redis
import os
from dotenv import load_dotenv

load_dotenv()


class RedisDBSingleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
        db_name = os.getenv("REDIS_DB_NAME")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db_name,
            decode_responses=True
        )

    def get_json(self, ids: list[UUID]):
        return [self.client.json().get(find_id) for find_id in ids]

