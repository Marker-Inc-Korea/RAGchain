import os
import pathlib
import pickle
from uuid import UUID

from langchain_core.runnables import RunnableLambda

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage, RetrievalResult

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)
query = "What is query decomposition?"


def test_load_passage():
    assert len(TEST_PASSAGES) > 0
    for passage in TEST_PASSAGES:
        assert isinstance(passage, Passage)
        assert isinstance(passage.id, UUID) or isinstance(passage.id, str)


def base_runnable_test(reranker: BaseReranker):
    runnable = reranker | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(query=query, passages=TEST_PASSAGES, scores=[]))
    assert len(result['passages']) == len(TEST_PASSAGES)
    assert result['passages'][0] != TEST_PASSAGES[0] or result['passages'][-1] != TEST_PASSAGES[-1]
    assert len(result['scores']) == len(result['passages'])
    assert isinstance(result['passages'][0], Passage)
    assert isinstance(result['scores'][0], float)
    for i in range(1, len(result['passages'])):
        assert result['scores'][i - 1] >= result['scores'][i]
