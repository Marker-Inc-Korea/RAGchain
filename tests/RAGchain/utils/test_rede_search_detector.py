import pytest

from RAGchain.benchmark.extra.search_detector_evaluator import SearchDetectorEvaluator
from RAGchain.utils.rede_search_detector import RedeSearchDetector

test_sentences = [
    'Does this hotel have rooms with a good view of the neighborhood?',
    'Are the portion sizes here large?',
    'Thanks so much for the train booking. I also need lodging with free wifi. I don\'t need free parking, though.'
]


@pytest.fixture
def rede_detector():
    detector = RedeSearchDetector()
    evaluator = SearchDetectorEvaluator(detector)
    evaluator.train()
    yield detector


def test_rede_detector(rede_detector):
    evaluator = SearchDetectorEvaluator(rede_detector)
    precision, recall, f1 = evaluator.evaluate()
    assert precision > 0.6
    assert recall > 0.6
    assert f1 > 0.6

    results = rede_detector.detect(test_sentences)
    assert results == [True, True, False]
