import logging
import os
import pathlib

import pytest

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import MSMARCOEvaluator
from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'ko_strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'ko_strategy_qa_evaluator.pkl')
logger = logging.getLogger(__name__)

@pytest.fixture
def msmarco_v1_1_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = BasicLLM(model_name='gpt-3.5-turbo-16k')
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = MSMARCOEvaluator(pipeline, evaluate_size=5,
                                 metrics=['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
                                          'answer_relevancy', 'faithfulness', 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG',
                                          'IndIDCG', 'IDCG', 'RR'],
                                 version='v1.1')

    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def msmarco_v2_1_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = BasicLLM(model_name='gpt-3.5-turbo-16k')
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = MSMARCOEvaluator(pipeline, evaluate_size=5,
                                 metrics=['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
                                          'answer_relevancy', 'faithfulness', 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG',
                                          'IndIDCG', 'IDCG', 'RR'],
                                 version='v2.1')

    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_msmarco_evaluator(msmarco_v1_1_evaluator, msmarco_v2_1_evaluator):
    result_v1_1 = msmarco_v1_1_evaluator.evaluate()

    assert len(result_v1_1.each_results) == 5
    assert result_v1_1.each_results.iloc[0]['question'] == 'does human hair stop squirrels'
    assert result_v1_1.each_results.iloc[0]['answer']
    for key, value in result_v1_1.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_v1_1.results)}")

    result_v2_1 = msmarco_v2_1_evaluator.evaluate()

    assert len(result_v2_1.each_results) == 5
    assert result_v2_1.each_results.iloc[0]['question'] == '. what is a corporation?'
    assert result_v2_1.each_results.iloc[0]['answer']
    for key, value in result_v2_1.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result_v2_1.results)}")
