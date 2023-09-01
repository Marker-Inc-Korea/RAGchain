import logging
import os

import pytest

import test_base_llm
from llm.rerank import RerankLLM
from utils.reranker import MonoT5Reranker

logger = logging.getLogger(__name__)
bm25_path = os.path.join(test_base_llm.root_dir, "resources", "bm25", "test_rerank_llm.pkl")
pickle_path = os.path.join(test_base_llm.root_dir, "resources", "pickle", "test_rerank_llm.pkl")


@pytest.fixture
def rerank_llm():
    test_base_llm.ready_pickle_db(pickle_path)
    retrieval = test_base_llm.ready_bm25_retrieval(bm25_path)
    reranker = MonoT5Reranker()
    llm = RerankLLM(retrieval=retrieval, reranker=reranker,
                    use_passage_count=4)
    yield llm
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_rerank_llm_ask(rerank_llm):
    answer, passages = rerank_llm.ask("What is reranker role?")
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)
