import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import BeirNQEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'ko_strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'ko_strategy_qa_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def beir_nq_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirNQEvaluator(pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_beir_nq_evaluator(beir_nq_evaluator):
    result = beir_nq_evaluator.evaluate()

    assert len(result.each_results) == 5
    assert result.each_results.iloc[0]['question'] == 'what is non controlling interest on balance sheet'
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result.results)}")
