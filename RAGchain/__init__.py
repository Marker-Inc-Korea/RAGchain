__version__ = '0.2.2'


# Sets the linker, which is required to use RAGchain.
import os
from dotenv import load_dotenv
from RAGchain.utils.linker import RedisDBSingleton, DynamoDBSingleton

load_dotenv()

linker_type = os.getenv("LINKER_TYPE")
if linker_type == "redisdb":
    linker = RedisDBSingleton()
elif linker_type == "dynamodb":
    linker = DynamoDBSingleton()
else:
    raise ValueError("Please set LINKER_TYPE to environment variable")
