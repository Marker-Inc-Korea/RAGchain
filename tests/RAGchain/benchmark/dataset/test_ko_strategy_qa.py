import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import KoStrategyQAEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'ko_strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'ko_strategy_qa_evaluator.pkl')


@pytest.fixture
def ko_strategy_qa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    pipeline = BasicRunPipeline(bm25_retrieval, OpenAI(model_name="babbage-002"))
    evaluator = KoStrategyQAEvaluator(pipeline, evaluate_size=5,
                                      metrics=['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score'])
    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_ko_strategy_qa_evaluator(ko_strategy_qa_evaluator):
    with pytest.raises(ValueError):
        ko_strategy_qa_evaluator.evaluate(validate_passages=True)

    result = ko_strategy_qa_evaluator.evaluate(validate_passages=False)
    assert len(result.each_results) == 5
    assert result.each_results.iloc[0]['question'] == '토마토 껍질을 벗기려면 뜨거운 물과 찬물이 모두 필요하나요?'
    assert result.each_results.iloc[0]['answer_pred']
    assert len(result.use_metrics) == len(ko_strategy_qa_evaluator.metrics)
