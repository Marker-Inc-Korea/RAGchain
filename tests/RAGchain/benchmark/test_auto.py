import logging
import os

import pytest

from RAGchain.DB import PickleDB
from RAGchain.benchmark import AutoEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval
from test_base import TEST_PASSAGES, root_dir

bm25_path = os.path.join(root_dir, "resources", "bm25", "auto_evaluator.pkl")
pickle_path = os.path.join(root_dir, "resources", "pickle", "auto_evaluator.pkl")

logger = logging.getLogger(__name__)


@pytest.fixture
def auto_evaluator():
    db = PickleDB(pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    retrieval = BM25Retrieval(bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    pipeline = BasicRunPipeline(retrieval=retrieval)
    yield AutoEvaluator(pipeline, questions=[
        "Where is the capital of France?",
        "Where is the largest city in Seoul?",
        "What is common between Seoul and Paris?"
    ])
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_auto_evaluator(auto_evaluator):
    result = auto_evaluator.evaluate()
    for key, res in result.results.items():
        assert res >= 0.0
        logger.info(f"{key}: {res}")

    assert len(result.each_results) == 3
    assert len(result.use_metrics) == 3
