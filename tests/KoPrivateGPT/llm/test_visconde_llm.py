import logging
import os

import pytest

import test_base_llm
from KoPrivateGPT.llm.visconde import ViscondeLLM

logger = logging.getLogger(__name__)
bm25_path = os.path.join(test_base_llm.root_dir, "resources", "bm25", "test_visconde_llm.pkl")
pickle_path = os.path.join(test_base_llm.root_dir, "resources", "pickle", "test_visconde_llm.pkl")


@pytest.fixture
def basic_llm():
    test_base_llm.ready_pickle_db(pickle_path)
    retrieval = test_base_llm.ready_bm25_retrieval(bm25_path)
    llm = ViscondeLLM(retrieval=retrieval, retrieve_size=20, use_passage_count=4)
    yield llm
    # teardown bm25
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    # teardown pickle
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_basic_llm_ask(basic_llm):
    answer, passages = basic_llm.ask("Is reranker and retriever have same role?")
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)
