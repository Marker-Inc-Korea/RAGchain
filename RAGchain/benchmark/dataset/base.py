
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.retrieval.base import BaseRetrieval


class BaseDatasetEvaluator(BaseEvaluator, ABC):
    @abstractmethod
    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB,
               ingest_size: Optional[int] = None):
        pass


class BaseStrategyQA:
    def convert_qa_to_pd(self, data):
        result = []
        for key, value in data.items():
            result.append([
                value['question'],
                value['answer'],
                self.__unpack_evidence(value['evidence'])
            ])
        return pd.DataFrame(result, columns=['question', 'answer', 'evidence'])

    def __unpack_evidence(self, evidence) -> List[str]:
        evidence_per_annotator = []
        for annotator in evidence:
            evidence_per_annotator.extend(
                list(set(
                    evidence_id
                    for step in annotator
                    for x in step
                    if isinstance(x, list)
                    for evidence_id in x
                ))
            )
        return evidence_per_annotator
