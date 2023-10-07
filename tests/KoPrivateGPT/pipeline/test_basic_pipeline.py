import logging
import os
import pathlib

import pytest

from KoPrivateGPT.DB import MongoDB
from KoPrivateGPT.pipeline.basic import BasicIngestPipeline, BasicRunPipeline
from KoPrivateGPT.preprocess.loader import FileLoader
from KoPrivateGPT.retrieval import BM25Retrieval

log = logging.getLogger(__name__)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")
bm25_path = os.path.join(root_dir, "resources", "bm25", "bm25_basic_pipeline.pkl")
mongodb_collection_name = "test_basic_pipeline"
mongodb_config = {
    "mongo_url": os.getenv('MONGO_URL'),
    "db_name": os.getenv('MONGO_DB_NAME'),
    "collection_name": mongodb_collection_name
}


@pytest.fixture
def basic_run_pipeline():
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    ingest_pipeline = BasicIngestPipeline(
        file_loader=FileLoader(file_dir, os.getenv('HWP_CONVERTER_HOST')),
        db=MongoDB(**mongodb_config),
        retrieval=BM25Retrieval(bm25_path)
    )
    ingest_pipeline.run()
    pipeline = BasicRunPipeline(
        retrieval=BM25Retrieval(bm25_path)
    )
    yield pipeline
    # teardown mongo db
    mongo_db = MongoDB(**mongodb_config)
    mongo_db.create_or_load()
    assert mongo_db.collection_name == mongodb_collection_name
    mongo_db.collection.drop()
    assert mongodb_collection_name not in mongo_db.db.list_collection_names()

    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)


def test_basic_pipeline(basic_run_pipeline):
    assert os.path.exists(bm25_path)
    query = "What is the purpose of KoPrivateGPT project? And what inspired it?"
    log.info(f"query: {query}")
    answer, passages = basic_run_pipeline.run(query=query)
    assert bool(answer) is True
    log.info(f"answer: {answer}")
    assert len(passages) > 0
    passage_str = "\n---------------\n".join([passage.content for passage in passages])
    log.info(f"passages: {passage_str}")
