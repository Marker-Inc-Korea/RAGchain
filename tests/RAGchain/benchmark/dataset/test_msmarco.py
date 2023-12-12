import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import MSMARCOEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
logger = logging.getLogger(__name__)

v1_1_bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'msmarco_v1_1_evaluator.pkl')
v1_1_pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'msmarco_v1_1_evaluator.pkl')

v2_1_bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'msmarco_v2_1_evaluator.pkl')
v2_1_pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'msmarco_v2_1_evaluator.pkl')


@pytest.fixture
def msmarco_v1_1_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=v1_1_bm25_path)
    db = PickleDB(v1_1_pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = MSMARCOEvaluator(pipeline, evaluate_size=5,
                                 version='v1.1')

    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(v1_1_bm25_path):
        os.remove(v1_1_bm25_path)
    if os.path.exists(v1_1_pickle_path):
        os.remove(v1_1_pickle_path)


@pytest.fixture
def msmarco_v2_1_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=v2_1_bm25_path)
    db = PickleDB(v2_1_pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = MSMARCOEvaluator(pipeline, evaluate_size=5,
                                 version='v2.1')

    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(v2_1_bm25_path):
        os.remove(v2_1_bm25_path)
    if os.path.exists(v2_1_pickle_path):
        os.remove(v2_1_pickle_path)


def test_msmarco_evaluator(msmarco_v1_1_evaluator, msmarco_v2_1_evaluator):
    result_v1_1 = msmarco_v1_1_evaluator.evaluate()

    assert len(result_v1_1.each_results) == 5
    assert result_v1_1.each_results.iloc[0]['question'] == 'does human hair stop squirrels'
    assert result_v1_1.each_results.iloc[0]['answer_pred']
    logger.info('The result of msmarco v1.1 dataset.')
    for key, value in result_v1_1.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_v1_1.results)}")

    result_v2_1 = msmarco_v2_1_evaluator.evaluate()

    assert len(result_v2_1.each_results) == 5
    assert result_v2_1.each_results.iloc[0]['question'] == '. what is a corporation?'
    assert result_v2_1.each_results.iloc[0]['answer_pred']
    logger.info('The result of msmarco v2.1 dataset.')
    for key, value in result_v2_1.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_v2_1.results)}")
