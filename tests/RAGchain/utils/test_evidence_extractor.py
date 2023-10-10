import logging

import pytest

from RAGchain.schema import Passage
from RAGchain.utils.evidence_extractor import EvidenceExtractor

logger = logging.getLogger(__name__)

TEST_PASSAGES = [
    Passage(
        content='Table TABREF19 and TABREF26 report zero-shot results on Europarl and Multi-UN evaluation sets, respectively. We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual transfer without pretraining BIBREF16. The results show that our approaches consistently outperform other approaches across languages and datasets, especially surpass pivoting, which is a strong baseline in the zero-shot scenario that multilingual NMT systems often fail to beat BIBREF19, BIBREF20, BIBREF23. Pivoting translates source to pivot then to target in two steps, causing inefficient translation process. Our approaches use one encoder-decoder model to translate between any zero-shot directions, which is more efficient than pivoting. Regarding the comparison between transfer approaches, our cross-lingual pretraining based transfer outperforms transfer method that does not use pretraining by a large margin.',
        filepath='test_filepath'),
    Passage(
        content="Regarding comparison between the baselines in table TABREF19, we find that pivoting is the strongest baseline that has significant advantage over other two baselines. Cross-lingual transfer for languages without shared vocabularies BIBREF16 manifests the worst performance because of not using source$\\leftrightarrow $pivot parallel data, which is utilized as beneficial supervised signal for the other two baselines.",
        filepath='test_filepath'),
    Passage(
        content='Our best approach of MLM+BRLM-SA achieves the significant superior performance to all baselines in the zero-shot directions, improving by 0.9-4.8 BLEU points over the strong pivoting. Meanwhile, in the supervised direction of pivot$\\rightarrow $target, our approaches performs even better than the original supervised Transformer thanks to the shared encoder trained on both large-scale monolingual data and parallel data between multiple languages.',
        filepath='test_filepath'),
    Passage(
        content='MLM alone that does not use source$\\leftrightarrow $pivot parallel data performs much better than the cross-lingual transfer, and achieves comparable results to pivoting. When MLM is combined with TLM or the proposed BRLM, the performance is further improved. MLM+BRLM-SA performs the best, and is better than MLM+BRLM-HA indicating that soft alignment is helpful than hard alignment for the cross-lingual pretraining.',
        filepath='test_filepath')
]


@pytest.fixture
def evidence_extractor():
    yield EvidenceExtractor()


def test_evidence_extractor(evidence_extractor):
    question = 'which multilingual approaches do they compare with?'
    evidence = evidence_extractor.extract(question, TEST_PASSAGES)
    logger.info(f'Evidence: {evidence}')
    assert bool(evidence) is True
    assert "We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual transfer without pretraining BIBREF16." in evidence
    irrelevant_question = 'what is the purpose of RAGchain project?'
    evidence = evidence_extractor.extract(irrelevant_question, TEST_PASSAGES)
    logger.info(f'Irrelevant Evidence: {evidence}')
    assert 'No Fragment' in evidence
