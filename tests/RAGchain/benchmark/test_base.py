import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval import BM25Retrieval
from RAGchain.schema import Passage

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
bm25_path = os.path.join(root_dir, "resources", "bm25", "base_evaluator.pkl")
pickle_path = os.path.join(root_dir, "resources", "pickle", "base_evaluator.pkl")

logger = logging.getLogger(__name__)

TEST_PASSAGES = [
    Passage(id='id-1',
            content='The capital of Korea is Seoul. And it is well-known.',
            filepath='./korea.txt'),
    Passage(id='id-2',
            content='The capital of France is Paris. And it is well-known.',
            filepath='./france.txt'),
    Passage(id='id-3',
            content='The capital of Germany is Berlin. And it is well-known.',
            filepath='./germany.txt'),
    Passage(id='id-4',
            content='The capital of Japan is Tokyo. And it is well-known.',
            filepath='./japan.txt'),
    Passage(id='id-5',
            content='The capital of China is Beijing. And it is well-known.',
            filepath='./china.txt'),
    Passage(id='id-6',
            content='The capital of Argentina is Buenos Aires.',
            filepath='./argentina.txt'),
    Passage(id='id-7',
            content='As of 2021, the largest city in Korea is Seoul.',
            filepath='./korea.txt'),
    Passage(id='id-8',
            content='As of 2021, the largest city in France is Paris.',
            filepath='./france.txt'),
    Passage(id='id-9',
            content='As of 2021, the largest city in Germany is Berlin.',
            filepath='./germany.txt'),
    Passage(id='id-10',
            content='As of 2021, the largest city in Japan is Tokyo.',
            filepath='./japan.txt'),
    Passage(id='id-11',
            content='As of 2021, the largest city in China is Beijing.',
            filepath='./china.txt')
]


class DummyEvaluator(BaseEvaluator):
    def __init__(self, pipeline: BaseRunPipeline, metrics=None, run_all=True):
        super().__init__(run_all=run_all, metrics=metrics)
        self.pipeline = pipeline

    def evaluate(self, **kwargs):
        questions = [
            "What is the capital of France?",
            "What is the capital of Korea?",
            "What is the capital of Japan?",
            "What is the capital of China?",
            "What is the capital of Germany?"
        ]
        retrieval_gt = [
            ['id-2', 'id-8'],
            ['id-1', 'id-7'],
            ['id-4'],
            ['id-5', 'id-11'],
            ['id-3']
        ]
        retrieval_gt_order = [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]
        ]
        answer_gt = [
            ['Paris'],
            ['Seoul'],
            ['Tokyo'],
            ['Beijing'],
            ['Berlin']
        ]

        return self._calculate_metrics(questions, self.pipeline,
                                       retrieval_gt=retrieval_gt,
                                       retrieval_gt_order=retrieval_gt_order,
                                       answer_gt=answer_gt,
                                       **kwargs)


@pytest.fixture
def dummy_evaluator():
    db = PickleDB(pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    retrieval = BM25Retrieval(bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    pipeline = BasicRunPipeline(retrieval=retrieval, llm=OpenAI())
    yield DummyEvaluator(pipeline)
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def no_ragas_evaluator():
    db = PickleDB(pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    retrieval = BM25Retrieval(bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    pipeline = BasicRunPipeline(retrieval=retrieval, llm=OpenAI())
    # test that it can initialize without openai api key env
    evaluator = DummyEvaluator(pipeline, metrics=['Recall', 'Precision', 'F1_score', 'BLEU'], run_all=False)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


# default top_k is 4
def test_base_evaluator(dummy_evaluator):
    result = dummy_evaluator.evaluate()
    for key, res in result.results.items():
        assert res >= 0.0
        logger.info(f"{key}: {res}")

    assert len(result.each_results) == 5
    assert result.each_results.iloc[0]['question'] == 'What is the capital of France?'
    assert result.each_results.iloc[0]['passage_contents'][0] == 'The capital of France is Paris. And it is well-known.'
    assert result.each_results.iloc[0]['passage_ids'][0] == 'id-2'
    assert result.each_results.iloc[0]['F1_score'] > 0
    assert len(result.use_metrics) == len(dummy_evaluator.metrics)

def test_no_ragas(no_ragas_evaluator):
    # It just tests it can initialize without ragas metrics.
    result = no_ragas_evaluator.evaluate()
    for key, res in result.results.items():
        assert res >= 0.0
        logger.info(f"{key}: {res}")

    assert len(result.each_results) == 5
    assert len(result.use_metrics) == len(no_ragas_evaluator.metrics)
