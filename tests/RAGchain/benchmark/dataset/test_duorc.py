import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import DuoRCEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
paraphrase_rc_bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'duorc_paraphrase_rc_evaluator.pkl')
paraphrase_rc_pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'duorc_paraphrase_rc_evaluator.pkl')

self_rc_bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'duorc_self_rc_evaluator.pkl')
self_rc_pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'duorc_self_rc_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def duorc_paraphrase_rc_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=paraphrase_rc_bm25_path)
    db = PickleDB(paraphrase_rc_pickle_path)
    llm = OpenAI(model_name="gpt-3.5-turbo-16k")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = DuoRCEvaluator(pipeline, evaluate_size=5, sub_dataset_name='ParaphraseRC')
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(paraphrase_rc_bm25_path):
        os.remove(paraphrase_rc_bm25_path)
    if os.path.exists(paraphrase_rc_pickle_path):
        os.remove(paraphrase_rc_pickle_path)


@pytest.fixture
def duorc_self_rc_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=self_rc_bm25_path)
    db = PickleDB(self_rc_pickle_path)
    llm = OpenAI(model_name="gpt-3.5-turbo-16k")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = DuoRCEvaluator(pipeline, evaluate_size=5, sub_dataset_name='SelfRC')
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(self_rc_bm25_path):
        os.remove(self_rc_bm25_path)
    if os.path.exists(self_rc_pickle_path):
        os.remove(self_rc_pickle_path)


def test_duorc_evaluator(duorc_paraphrase_rc_evaluator,
                         duorc_self_rc_evaluator):
    result_paraphrase_rc = duorc_self_rc_evaluator.evaluate()
    result_self_rc = duorc_self_rc_evaluator.evaluate()

    assert len(result_paraphrase_rc.each_results) == 5
    assert len(result_self_rc.each_results) == 5

    logger.info("The result of ParaphraseRC")
    for key, value in result_paraphrase_rc.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_paraphrase_rc.results)}")

    logger.info("---------------------------------------------------------------")
    logger.info("The result of SelfRC")
    for key, value in result_self_rc.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_self_rc.results)}")
