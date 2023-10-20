from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class QasperEvaluator(BaseDatasetEvaluator):
    dataset_name = "allenai/qasper"

    def __init__(self, run_pipeline: BasePipeline, evaluate_size: int, metrics: Optional[List[str]] = None):
        support_metrics = []
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)
        self.run_pipeline = run_pipeline
        self.test_data = load_dataset(self.dataset_name)['test'].to_pandas()
        if len(self.test_data) > evaluate_size:
            self.test_data = self.test_data.sample(evaluate_size)
        self.test_data = self.preprocess(self.test_data)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        if ingest_size is not None:
            raise Warning("QasperEvaluator does not support ingest_size parameter. "
                          "You can adjust evaluate_size parameter in __init__ method.")

        passages = self.test_data['passages'].tolist()
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self) -> EvaluateResult:
        pass

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

        def find_passage_id(passages: List[Passage], content: str):
            if content.startswith('FLOAT SELECTED: '):
                content = content.replace('FLOAT SELECTED: ', '')
            filtered_passages = list(filter(lambda x: x.content == content, passages))
            if len(filtered_passages) <= 0:
                return None
            return filtered_passages[0].id

        def make_question(row):
            return row['qas']['question']

        def make_retrieval_gt(row):
            result = list(set([
                find_passage_id(row['passages'], evidence)
                for answer_group in row['qas']['answers']
                for answer in answer_group['answer']
                for evidence in answer['evidence']
            ]))
            return [elem for elem in result if elem is not None]

        def make_answer_gt(row):
            for answer_group in row['qas']['answers']:
                for answer in answer_group['answer']:
                    if answer['free_form_answer'] != '':
                        return answer['free_form_answer']
            return None

        data['question'] = data.apply(make_question, axis=1)
        data['retrieval_gt'] = data.apply(make_retrieval_gt, axis=1)
        data['answer_gt'] = data.apply(make_answer_gt, axis=1)
        return data
