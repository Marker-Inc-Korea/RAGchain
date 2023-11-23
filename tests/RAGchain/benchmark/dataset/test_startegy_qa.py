import os
import pathlib

import pytest

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import StrategyQAEvaluator
from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'strategy_qa_evaluator.pkl')


@pytest.fixture
def strategy_qa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = BasicLLM(model_name='gpt-3.5-turbo-16k')
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = StrategyQAEvaluator(pipeline, evaluate_size=5,
                                    metrics=['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score'])
    evaluator.ingest([bm25_retrieval], db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_ko_strategy_qa_evaluator(strategy_qa_evaluator):
    with pytest.raises(ValueError):
        strategy_qa_evaluator.evaluate(validate_passages=True)

    result = strategy_qa_evaluator.evaluate(validate_passages=False)
    assert len(result.each_results) == 5
    assert result.each_results.iloc[0][
               'question'] == 'Are more people today related to Genghis Khan than Julius Caesar?'
    assert result.each_results.iloc[0]['answer_pred']
    assert len(result.use_metrics) == len(strategy_qa_evaluator.metrics)
