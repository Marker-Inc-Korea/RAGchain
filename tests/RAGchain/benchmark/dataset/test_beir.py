import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI

from RAGchain.DB import PickleDB
from RAGchain.benchmark.dataset import (BeirFEVEREvaluator, BeirFIQAEvaluator,
                                        BeirHOTPOTQAEvaluator,
                                        BeirQUORAEvaluator, BeirSCIDOCSEvaluator,
                                        BeirSCIFACTEvaluator)
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
logger = logging.getLogger(__name__)


# pkl file name must be different to each test evaluators.
def generate_path(evaluator_name: str = None):
    valid_evaluator = ['fever', 'fiqa', 'hotpotqa', 'quora', 'scidocs', 'scifact']
    if evaluator_name is None:
        raise ValueError("You don't input evaluator name.")
    if evaluator_name.lower() not in valid_evaluator:
        raise ValueError(f"You input {evaluator_name} that is invalid evaluator.")

    bm25_path = os.path.join(root_dir, 'resources', 'bm25', f'beir_{evaluator_name}_evaluator.pkl')
    pickle_path = os.path.join(root_dir, 'resources', 'pickle', f'beir_{evaluator_name}_evaluator.pkl')
    return bm25_path, pickle_path


# This function set default values using all beir datasets evaluators.
# Set default retrieval, db, and llm.
def set_beir_evaluator(bm25_path, pickle_path):
    retrieval = BM25Retrieval(save_path=bm25_path)
    db = PickleDB(pickle_path)
    llm = OpenAI(model_name="gpt-3.5-turbo-16k")
    pipeline = BasicRunPipeline(retrieval, llm)
    return retrieval, db, llm, pipeline


def remove_path(bm25_path, pickle_path):
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


@pytest.fixture
def beir_fever_evaluator():
    bm25_path, pickle_path = generate_path('fever')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirFEVEREvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


@pytest.fixture
def beir_fiqa_evaluator():
    bm25_path, pickle_path = generate_path('fiqa')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirFIQAEvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


@pytest.fixture
def beir_hotpotqa_evaluator():
    bm25_path, pickle_path = generate_path('hotpotqa')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirHOTPOTQAEvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


@pytest.fixture
def beir_quora_evaluator():
    bm25_path, pickle_path = generate_path('quora')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirQUORAEvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


@pytest.fixture
def beir_scidocs_evaluator():
    bm25_path, pickle_path = generate_path('scidocs')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirSCIDOCSEvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


@pytest.fixture
def beir_scifact_evaluator():
    bm25_path, pickle_path = generate_path('scifact')
    retrieval, db, llm, pipeline = set_beir_evaluator(bm25_path, pickle_path)
    evaluator = BeirSCIFACTEvaluator(run_pipeline=pipeline, evaluate_size=5)
    evaluator.ingest(retrievals=[retrieval], db=db, ingest_size=20)

    yield evaluator
    remove_path(bm25_path, pickle_path)


def test_beir_evaluator(beir_fever_evaluator,
                        beir_fiqa_evaluator,
                        beir_hotpotqa_evaluator,
                        beir_quora_evaluator,
                        beir_scidocs_evaluator,
                        beir_scifact_evaluator
                        ):
    fever_result = beir_fever_evaluator.evaluate()
    fiqa_result = beir_fiqa_evaluator.evaluate()
    hotpotqa_result = beir_hotpotqa_evaluator.evaluate()
    quora_result = beir_quora_evaluator.evaluate()
    scidocs_result = beir_scidocs_evaluator.evaluate()
    scifact_result = beir_scifact_evaluator.evaluate()

    # Assertion of fever
    logger.info('Result of fever')
    assert fever_result.each_results.iloc[0]['question'] == 'Julie Bowen was born on March 3.'
    validation_result(fever_result)

    # Assertion of fiqa
    logger.info('Result of fiqa')
    assert fiqa_result.each_results.iloc[0]['question'] == 'Why diversify stocks/investments?'
    validation_result(fiqa_result)

    # Assertion of hotpotqa
    logger.info('Result of hotpotqa')
    assert hotpotqa_result.each_results.iloc[0][
               'question'] == 'Which piece did Ludwig van Beethoven publish in 1801 that was dedicated to Count Moritz von Fries?'
    validation_result(hotpotqa_result)

    # Assertion of quora
    logger.info('Result of quora')
    assert quora_result.each_results.iloc[0][
               'question'] == 'Why is the melting point of diamond so much higher than silicon?'
    validation_result(quora_result)

    # Assertion of scidocs
    logger.info('Result of scidocs')
    assert scidocs_result.each_results.iloc[0][
               'question'] == 'Bank distress in the news: Describing events through deep learning'
    validation_result(scidocs_result)

    # Assertion of scifact
    logger.info('Result of scifact')
    assert scifact_result.each_results.iloc[0][
               'question'] == '0-dimensional biomaterials show inductive properties.'
    validation_result(scifact_result)


def validation_result(result):
    assert len(result.each_results) == 5
    for key, value in result.results.items():
        logger.info(f"{key}: {value}")
    logger.info("The result length is " + f"{len(result.results)}")
    logger.info("----------------------------------------------------------------")
