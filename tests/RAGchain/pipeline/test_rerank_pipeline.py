import logging
import os
import pathlib
import pickle

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.pipeline import RerankRunPipeline
from RAGchain.reranker import TARTReranker
from RAGchain.retrieval import BM25Retrieval

logger = logging.getLogger(__name__)
root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
bm25_path = os.path.join(root_dir, "resources", "bm25", "bm25_rerank_pipeline.pkl")
pickle_path = os.path.join(root_dir, "resources", "pickle", "pickle_rerank_pipeline.pkl")
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)


@pytest.fixture
def rerank_run_pipeline():
    # ingest files
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    retrieval = BM25Retrieval(save_path=bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    reranker = TARTReranker("Find passage to answer given question")
    pipeline = RerankRunPipeline(retrieval, reranker, OpenAI(model_name="babbage-002"), use_passage_count=4)
    yield pipeline
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_rerank_run_pipeline(rerank_run_pipeline):
    answer, passages, scores = rerank_run_pipeline.get_passages_and_run(["What is reranker role?",
                                                                         "What is the purpose of reranker?"])
    logger.info(f"Answer: {answer[0]}")
    assert bool(answer[0])
    assert len(answer) == len(passages) == len(scores) == 2
    assert len(passages[0]) == len(scores[0]) == 4
    for i in range(1, len(scores[0])):
        assert scores[0][i - 1] >= scores[0][i]

    result = rerank_run_pipeline.run.invoke(("What is reranker role?", 3))
    logger.info(f"Answer: {result}")
    assert bool(result)
    assert isinstance(result, str)
