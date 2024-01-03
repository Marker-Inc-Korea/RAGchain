import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import SearchQAEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'search_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'search_qa_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def search_qa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = SearchQAEvaluator(pipeline, evaluate_size=5)
    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_search_qa_evaluator(search_qa_evaluator):
    result = search_qa_evaluator.evaluate()

    assert len(result.each_results) == 5
    assert result.each_results.iloc[0][
               'question'] == "Title of a Robert Louis Stevenson book about a boy's abduction"
    assert result.each_results.iloc[0]['answer_pred']
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result.results)}")
