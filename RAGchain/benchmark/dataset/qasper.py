import itertools
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class QasperEvaluator(BaseDatasetEvaluator):
    dataset_name = "NomaDamas/qasper"

    def __init__(self, run_pipeline: BasePipeline, evaluate_size: int, metrics: Optional[List[str]] = None):
        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision', 'answer_relevancy', 'faithfulness']
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)
        self.run_pipeline = run_pipeline
        self.data = load_dataset(self.dataset_name)['train'].to_pandas()
        self.data = self.data.drop('__index_level_0__', axis=1)
        if len(self.data) > evaluate_size:
            self.data = self.data.sample(evaluate_size)
        self.data = self.preprocess(self.data)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        if ingest_size is not None:
            raise Warning("QasperEvaluator does not support ingest_size parameter. "
                          "You can adjust evaluate_size parameter in __init__ method.")

        passages = list(itertools.chain(*self.data['passages'].tolist()))
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self) -> EvaluateResult:
        return self._calculate_metrics(
            questions=list(itertools.chain(*self.data['question'].tolist())),
            pipeline=self.run_pipeline,
            retrieval_gt=list(itertools.chain(*self.data['retrieval_gt'].tolist())),
            answer_gt=list(itertools.chain(*self.data['answer_gt'].tolist()))
        )

    def preprocess(self, data):
        def make_passages(row):
            # convert full_text to passages
            passages = [
                Passage(
                    id=f'{row["id"]}-{section_name}-{i}',
                    content=paragraph,
                    filepath=self.dataset_name,
                    metadata_etc={'doi': row['id'], 'section_name': section_name}
                )
                for section_name, paragraphs in zip(row['full_text']['section_name'], row['full_text']['paragraphs'])
                for i, paragraph in enumerate(paragraphs)
            ]

            passages += [
                Passage(
                    id=f'{row["id"]}-{file}',
                    content=caption,
                    filepath=self.dataset_name,
                    metadata_etc={'doi': row['id'], 'file': file}
                )
                for caption, file in zip(row['figures_and_tables']['caption'], row['figures_and_tables']['file'])
            ]

            return passages

        data['passages'] = data.apply(make_passages, axis=1)
        return data
