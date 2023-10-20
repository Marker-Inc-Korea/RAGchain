from abc import ABC, abstractmethod
from typing import List

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.retrieval.base import BaseRetrieval


class BaseDatasetEvaluator(BaseEvaluator, ABC):
    @abstractmethod
    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB):
        pass

    @staticmethod
    def download_hf_dataset(dataset_name: str):
        """
        Download dataset from huggingface
        """
        return load_dataset(dataset_name)
