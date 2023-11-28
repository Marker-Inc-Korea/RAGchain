import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import QasperEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'strategy_qa_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def qasper_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = QasperEvaluator(pipeline, evaluate_size=2)
    evaluator.ingest([bm25_retrieval], db)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_qasper_evaluator(qasper_evaluator):
    result = qasper_evaluator.evaluate()
    assert len(result.use_metrics) == len(qasper_evaluator.metrics)

    def same_id(row):
        same_prefix = (
                row.passage_ids[0][:10] == row.passage_ids[1][:10]
                == row.passage_ids[2][:10] == row.passage_ids[3][:10]
        )
        return same_prefix

    same_id_count = result.each_results.apply(same_id, axis=1).sum()
    assert len(result.each_results) == 2
    assert result.each_results.iloc[0, 0] == 'What evaluation metric is used?'
    assert same_id_count == len(result.each_results)
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
