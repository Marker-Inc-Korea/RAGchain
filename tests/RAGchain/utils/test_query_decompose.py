import logging
from typing import List

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.utils.query_decompose import QueryDecomposition

logger = logging.getLogger(__name__)

query = "Is Disneyland in Paris the largest Disneyland in the world?"


@pytest.fixture
def query_decompose():
    yield QueryDecomposition(OpenAI(temperature=0.2))


def test_query_decompose(query_decompose):
    result = query_decompose.decompose(query)
    check_decompose(result)


def test_query_decompose_runnable(query_decompose):
    result = query_decompose.invoke(query)
    check_decompose(result)


def check_decompose(result: List[str]):
    logger.info(f"result : {result}")
    assert len(result) > 1
    for res in result:
        assert isinstance(res, str)
        assert bool(res)
