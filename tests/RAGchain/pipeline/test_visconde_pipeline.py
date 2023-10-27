import logging
import os
import pathlib
import pickle

import pytest

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
    pipeline = ViscondeRunPipeline(retrieval)
    yield pipeline
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_rerank_run_pipeline(visconde_run_pipeline):
    answer, passages = visconde_run_pipeline.run("Is reranker and retriever have same role?")
    logger.info(f"Answer: {answer}")
    assert bool(answer)
    assert len(passages) == 3
