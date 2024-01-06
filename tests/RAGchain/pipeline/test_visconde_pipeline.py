import logging
import os
import pathlib
import pickle

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.pipeline import ViscondeRunPipeline
from RAGchain.retrieval import BM25Retrieval

logger = logging.getLogger(__name__)
root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
bm25_path = os.path.join(root_dir, "resources", "bm25", "bm25_visconde_pipeline.pkl")
pickle_path = os.path.join(root_dir, "resources", "pickle", "pickle_visconde_pipeline.pkl")
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)


@pytest.fixture
def visconde_run_pipeline():
    # ingest files
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    retrieval = BM25Retrieval(save_path=bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    pipeline = ViscondeRunPipeline(retrieval, OpenAI(model_name="babbage-002", temperature=0.1), use_passage_count=4)
    yield pipeline
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_visconde_run_pipeline(visconde_run_pipeline):
    answer = visconde_run_pipeline.run.invoke("Is reranker and retriever have same role?")
    logger.info(f"Answer: {answer}")
    assert bool(answer)


def test_visconde_run_pipeline_get_passages_and_run(visconde_run_pipeline):
    answers, passages, scores = visconde_run_pipeline.get_passages_and_run(["Is reranker and retriever have same role?",
                                                                            "What is reranker role?"], top_k=70)
    logger.info(f"Answer 1: {answers[0]}")
    logger.info(f"Answer 2: {answers[1]}")
    assert len(answers) == len(passages) == len(scores) == 2
    assert len(passages[0]) == len(scores[0]) == 4
