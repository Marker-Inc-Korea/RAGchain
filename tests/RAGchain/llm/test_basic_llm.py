import logging
import os
from typing import List

import pytest

import test_base_llm
from RAGchain.llm.basic import BasicLLM
from RAGchain.schema import Passage

logger = logging.getLogger(__name__)
bm25_path = os.path.join(test_base_llm.root_dir, "resources", "bm25", "test_basic_llm.pkl")
pickle_path = os.path.join(test_base_llm.root_dir, "resources", "pickle", "test_basic_llm.pkl")


def basic_en_prompt(passages: List[Passage], question: str) -> List[dict]:
    context = "\n\n".join([passage.content for passage in passages])
    return [
        {"role": "system", "content": "Please answer the question based on the given documents."},
        {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {question}\n"},
        {"role": "assistant", "content": "The following is an answer to the question."}
    ]


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
def basic_llm():
    llm = BasicLLM(prompt_func=basic_en_prompt, stream_func=lambda x: logger.info(x))
    yield llm


def simple_llm_run(query, retrieval, llm, top_k: int = 5, *args, **kwargs):
    passages = retrieval.retrieve(query, top_k=top_k)
    return llm.ask(query, passages, *args, **kwargs)


def test_basic_llm_ask(basic_llm, bm25_retrieval):
    answer, passages = simple_llm_run("What is reranker role?", bm25_retrieval, basic_llm)
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)
    answer, passages = simple_llm_run("What is retriever role?", bm25_retrieval, basic_llm, top_k=3)
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages, passage_cnt=3)


def test_basic_llm_ask_stream(basic_llm, bm25_retrieval):
    answer, passages = simple_llm_run("What is reranker role?", bm25_retrieval, basic_llm, stream=True)
    logger.info(f"Answer: {answer}")
    test_base_llm.validate_answer(answer, passages)


def test_basic_llm_chat_history(basic_llm, bm25_retrieval):
    answer, passages = simple_llm_run("What is reranker role?", bm25_retrieval, basic_llm)
    assert basic_llm.chat_history[0] == {"role": "user", "content": "What is reranker role?"}
    assert basic_llm.chat_history[1] == {"role": "assistant", "content": answer}
    simple_llm_run("What is retriever role?", bm25_retrieval, basic_llm)
    assert len(basic_llm.chat_history) == 4
    simple_llm_run("What is llm role?", bm25_retrieval, basic_llm)
    simple_llm_run("What is reranker role?", bm25_retrieval, basic_llm)
    assert basic_llm.chat_history[-basic_llm.chat_offset:][0] == {"role": "user", "content": "What is retriever role?"}
    store_chat_history = basic_llm.clear_chat_history()
    assert len(basic_llm.chat_history) == 0
    assert len(store_chat_history) == 8
