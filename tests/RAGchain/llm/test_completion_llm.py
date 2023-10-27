import logging
import os

import pytest

import test_base_llm
from RAGchain.llm.completion import CompletionLLM

logger = logging.getLogger(__name__)
bm25_path = os.path.join(test_base_llm.root_dir, "resources", "bm25", "test_completion_llm.pkl")
pickle_path = os.path.join(test_base_llm.root_dir, "resources", "pickle", "test_completion_llm.pkl")


@pytest.fixture
def bm25_retrieval():
    test_base_llm.ready_pickle_db(pickle_path)
    retrieval = test_base_llm.ready_bm25_retrieval(bm25_path)
    yield retrieval
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def completion_llm():
    llm = CompletionLLM(stream_func=lambda x: logger.info(x))
    yield llm


def test_completion_llm_ask(completion_llm, bm25_retrieval):
    answer, passages = test_base_llm.simple_llm_run(
        "Is reranker and retriever have same role?", bm25_retrieval, completion_llm, top_k=5)
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)


def test_completion_llm_ask_stream(completion_llm, bm25_retrieval):
    answer, passages = test_base_llm.simple_llm_run(
        "Is reranker and retriever have same role?", bm25_retrieval, completion_llm, top_k=5, stream=True)
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)
