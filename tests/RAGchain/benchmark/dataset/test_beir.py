import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import BeirEvaluator
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, 'resources', 'bm25', 'ko_strategy_qa_evaluator.pkl')
pickle_path = os.path.join(root_dir, 'resources', 'pickle', 'ko_strategy_qa_evaluator.pkl')
logger = logging.getLogger(__name__)


@pytest.fixture
def beir_fever_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='fever', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_fiqa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='fiqa', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_hotpotqa_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='hotpotqa', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_nq_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='nq', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_quora_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='quora', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_scidocs_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='scidocs', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_scifact_evaluator():
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="babbage-002")
    pipeline = BasicRunPipeline(bm25_retrieval, llm)
    evaluator = BeirEvaluator(run_pipeline=pipeline, file_name='scifact', evaluate_size=5)
    evaluator.ingest(retrievals=[bm25_retrieval], db=db, ingest_size=20)
    yield evaluator
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_beir_evaluator(beir_fever_evaluator,
                        beir_fiqa_evaluator,
                        beir_hotpotqa_evaluator,
                        beir_nq_evaluator,
                        beir_quora_evaluator,
                        beir_scidocs_evaluator,
                        beir_scifact_evaluator
                        ):
    fever_result = beir_fever_evaluator.evaluate()
    fiqa_result = beir_fiqa_evaluator.evaluate()
    hotpotqa_result = beir_hotpotqa_evaluator.evaluate()
    nq_result = beir_nq_evaluator.evaluate()
    quora_result = beir_quora_evaluator.evaluate()
    scidocs_result = beir_scidocs_evaluator.evaluate()
    scifact_result = beir_scifact_evaluator.evaluate()

    assertion_result = AssertResult(fever_result,
                                    fiqa_result,
                                    hotpotqa_result,
                                    nq_result,
                                    quora_result,
                                    scidocs_result,
                                    scifact_result
                                    )
    assertion_result.return_evaluation_results(fever_result,
                                               fiqa_result,
                                               hotpotqa_result,
                                               nq_result,
                                               quora_result,
                                               scidocs_result,
                                               scifact_result
                                               )


class AssertResult:
    def __init__(self, fever_result=None,
                 fiqa_result=None,
                 hotpotqa_result=None,
                 nq_result=None,
                 quora_result=None,
                 scidocs_result=None,
                 scifact_result=None
                 ):
        try:
            results_lst = [fever_result, fiqa_result,
                           hotpotqa_result, nq_result, quora_result,
                           scidocs_result, scifact_result]
        except:
            raise ValueError("Result is not input.")

    @staticmethod
    def return_evaluation_results(fever_result=None,
                                  fiqa_result=None,
                                  hotpotqa_result=None,
                                  nq_result=None,
                                  quora_result=None,
                                  scidocs_result=None,
                                  scifact_result=None):
        if fever_result is not None:
            AssertResult.fever_validation(fever_result)
        if fiqa_result is not None:
            AssertResult.fiqa_validation(fiqa_result)
        if hotpotqa_result is not None:
            AssertResult.hotpotqa_validation(hotpotqa_result)
        if nq_result is not None:
            AssertResult.nq_question_validation(nq_result)
        if quora_result is not None:
            AssertResult.quora_question_validation(quora_result)
        if scidocs_result is not None:
            AssertResult.scidocs_question_validation(scidocs_result)
        if scifact_result is not None:
            AssertResult.scifact_question_validation(scifact_result)

    @staticmethod
    def validate_results(result):
        assert len(result.each_results) == 5
        for key, value in result.results.items():
            logger.info(f"{key}: {value}")
        logger.info("The result length is " + f"{len(result.results)}")
        logger.info("----------------------------------------------------------------")

    @staticmethod
    def fever_validation(fever_result):
        logger.info('Result of fever')
        assert fever_result.each_results.iloc[0]['question'] == 'Julie Bowen was born on March 3.'
        AssertResult.validate_results(fever_result)

    @staticmethod
    def fiqa_validation(fiqa_result):
        logger.info('Result of fiqa')
        assert fiqa_result.each_results.iloc[0]['question'] == 'Why diversify stocks/investments?'
        AssertResult.validate_results(fiqa_result)

    @staticmethod
    def hotpotqa_validation(hotpotqa_result):
        logger.info('Result of hotpotqa')
        assert hotpotqa_result.each_results.iloc[0][
                   'question'] == 'Which piece did Ludwig van Beethoven publish in 1801 that was dedicated to Count Moritz von Fries?'
        AssertResult.validate_results(hotpotqa_result)

    @staticmethod
    def nq_question_validation(nq_result):
        logger.info('Result of nq')
        assert nq_result.each_results.iloc[0]['question'] == 'what is non controlling interest on balance sheet'
        AssertResult.validate_results(nq_result)

    @staticmethod
    def quora_question_validation(quora_result):
        logger.info('Result of quora')
        assert quora_result.each_results.iloc[0][
                   'question'] == 'Why is the melting point of diamond so much higher than silicon?'
        AssertResult.validate_results(quora_result)

    @staticmethod
    def scidocs_question_validation(scidocs_result):
        logger.info('Result of scidocs')
        assert scidocs_result.each_results.iloc[0][
                   'question'] == 'Bank distress in the news: Describing events through deep learning'
        AssertResult.validate_results(scidocs_result)

    @staticmethod
    def scifact_question_validation(scifact_result):
        logger.info('Result of scifact')
        assert scifact_result.each_results.iloc[0][
                   'question'] == '0-dimensional biomaterials show inductive properties.'
        AssertResult.validate_results(scifact_result)
