import logging
import os

import pytest

import test_base_llm
from llm.basic import BasicLLM

logger = logging.getLogger(__name__)
bm25_path = os.path.join(test_base_llm.root_dir, "resources", "bm25", "test_basic_llm.pkl")
pickle_path = os.path.join(test_base_llm.root_dir, "resources", "pickle", "test_basic_llm.pkl")


@pytest.fixture
def basic_llm():
    test_base_llm.ready_pickle_db(pickle_path)
    retrieval = test_base_llm.ready_bm25_retrieval(bm25_path)
    llm = BasicLLM(retrieval=retrieval)
    yield llm
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_ask(basic_llm):
    answer, passages = basic_llm.ask("What is reranker role?")
    assert bool(answer)
    logger.info(f"Answer: {answer}")
    assert len(passages) == 4

    solution_ids = [passage.id for passage in test_base_llm.TEST_PASSAGES]
    for passage in passages:
        assert passage.id in solution_ids
