import logging

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.utils.query_decompose import QueryDecomposition

logger = logging.getLogger(__name__)


@pytest.fixture
def query_decompose():
    yield QueryDecomposition(OpenAI())


def test_query_decompose(query_decompose):
    query = "Is Disneyland in Paris the largest Disneyland in the world?"
    result = query_decompose.decompose(query)
    logger.info(f"result : {result}")
    assert len(result) > 1
    for res in result:
        assert isinstance(res, str)
        assert bool(res)
