import os
import pathlib
from chromadb.config import Settings


class Options(object):
    root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
    source_dir = os.path.join(root_dir, "SOURCE_DOCUMENTS")
    embedded_files_cache_dir = os.path.join(root_dir, "embedded_files_cache.pkl")


class ChromaOptions(object):
    persist_dir = os.path.join(Options.root_dir, "DB")
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_dir,
        anonymized_telemetry=False
    )


class PineconeOptions(object):
    index_name = "ko-private-gpt"
