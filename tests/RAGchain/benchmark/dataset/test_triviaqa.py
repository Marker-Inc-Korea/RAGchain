import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import TriviaQAEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'triviaqa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'triviaqa_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def triviaqa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="gpt-3.5-turbo-16k")
    pipeline = BasicRunPipeline(bm25_retrieval, llm, retrieval_option={'top_k': 2})
    evaluator = TriviaQAEvaluator(pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_triviaqa_evaluator(triviaqa_evaluator):
    result = triviaqa_evaluator.evaluate()

    assert len(result.each_results) == 5
    assert result.each_results.iloc[0][
               'question'] == 'Who was the man behind The Chipmunks?'
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result.results)}")
