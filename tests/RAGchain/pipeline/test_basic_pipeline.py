import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import MongoDB
from RAGchain.pipeline.basic import BasicIngestPipeline, BasicRunPipeline, BasicRunPipelineNew
from RAGchain.preprocess.loader import FileLoader
from RAGchain.retrieval import BM25Retrieval

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


@pytest.fixture
def basic_run_pipeline_new():
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    ingest_pipeline = BasicIngestPipeline(
        file_loader=FileLoader(file_dir, os.getenv('HWP_CONVERTER_HOST')),
        db=MongoDB(**mongodb_config),
        retrieval=BM25Retrieval(bm25_path)
    )
    ingest_pipeline.run()
    pipeline = BasicRunPipelineNew(
        retrieval=BM25Retrieval(bm25_path),
        model=OpenAI()
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
    query = "What is the purpose of RAGchain project? And what inspired it?"
    log.info(f"query: {query}")
    answer, passages = basic_run_pipeline.run(query=query)
    assert bool(answer) is True
    log.info(f"answer: {answer}")
    assert len(passages) > 0
    passage_str = "\n---------------\n".join([passage.content for passage in passages])
    log.info(f"passages: {passage_str}")


def test_basic_pipeline_new(basic_run_pipeline_new):
    assert os.path.exists(bm25_path)
    query = "What is the purpose of RAGchain project? And what inspired it?"
    log.info(f"query: {query}")
    answer = basic_run_pipeline_new.invoke({"question": query})
    assert bool(answer) is True
    log.info(f"answer: {answer}")

    queries = ["What is the purpose of KoPrivateGPT project?",
               "What inspired KoPrivateGPT project?",
               "How can I install KoPrivateGPT project?"]
    answers, passages, scores = basic_run_pipeline_new.run(queries)
    assert len(answers) == len(queries)
    assert len(passages) == len(queries)
    assert len(scores) == len(queries)
    for query, answer, passage, score in zip(queries, answers, passages, scores):
        assert bool(answer) is True
        log.info(f"question: {query}\nanswer: {answer}")
        assert len(passage) == 5
        log.info(f"score: {score}")
