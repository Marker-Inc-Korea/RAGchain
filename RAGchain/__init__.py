__version__ = '0.2.6'

# Sets the linker, which is required to use RAGchain.
import os

from RAGchain.utils.linker import RedisLinker, DynamoLinker, JsonLinker

linker_type = os.getenv("LINKER_TYPE")
if linker_type == "redisdb":
    linker = RedisLinker()
elif linker_type == "dynamodb":
    linker = DynamoLinker()
elif linker_type == "json":
    linker = JsonLinker()
else:
    raise ValueError("Please set LINKER_TYPE to environment variable")
