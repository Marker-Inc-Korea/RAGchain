import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import DSTC11Evaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
logger = logging.getLogger(__name__)

bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'dstc_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'dstc_evaluator.pkl')


@pytest.fixture
def dstc_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = DSTC11Evaluator(pipeline, evaluate_size=5)

    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_dstc_evaluator(dstc_evaluator):
    result = dstc_evaluator.evaluate()

    assert len(result.each_results) == 5
    assert result.each_results.iloc[0]['question'] == ("U: I'm looking to stay at a 3 star hotel in the north. "
                                                       "S: Sorry, I have no results for that query. Would you like to "
                                                       "try a different area of town? U: Are there any moderate priced "
                                                       "hotels in the North? S: Yes I have two. Would you like me to book "
                                                       "one? U: I need a hotel to include free parking; does either have"
                                                       " that? S: Yes both of them have free parking. U: Which one would"
                                                       " you recommend? S: How about the Ashley hotel? U: Is the Ashley "
                                                       "hotel a 3 star hotel? S: the ashley is actually a 2 star hotel. "
                                                       "U: Does this hotel have rooms with a good view of the neighborhood?")
    assert result.each_results.iloc[0]['answer_pred']
    logger.info('The result of DSTC-11-Track-5 dataset.')
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result.results)}")
