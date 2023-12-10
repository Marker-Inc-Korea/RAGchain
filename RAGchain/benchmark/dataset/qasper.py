from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class QasperEvaluator(BaseDatasetEvaluator):
    """
    QasperEvaluator is a class for evaluating pipeline performance on Qasper dataset.
    """
    dataset_name = "NomaDamas/qasper"

    def __init__(self,
                 run_pipeline: BaseRunPipeline,
                 evaluate_size: int,
                 metrics: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        :param run_pipeline: pipeline to evaluate
        :param evaluate_size: number of data to evaluate
        :param metrics: metrics to evaluate. Default metrics are ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM',
        'F1_score', 'context_recall', 'context_precision', 'answer_relevancy', 'faithfulness']
        If None, use default metrics.
        :param random_state: random seed for sampling data. Default is 42.

        Notice:
        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """
        default_metrics = (self.retrieval_gt_metrics + self.answer_gt_metrics +
                           self.answer_no_gt_ragas_metrics + self.answer_passage_metrics)
        support_metrics = (self.retrieval_gt_metrics + self.retrieval_gt_ragas_metrics +
                           self.retrieval_no_gt_ragas_metrics + self.answer_gt_metrics +
                           self.answer_no_gt_ragas_metrics + self.answer_passage_metrics)
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)
        self.run_pipeline = run_pipeline
        self.data = load_dataset(self.dataset_name)['train'].to_pandas()
        self.data = self.data.drop('__index_level_0__', axis=1)
        if len(self.data) > evaluate_size:
            self.data = self.data.sample(evaluate_size, random_state=random_state)
        self.data = self.preprocess(self.data)

        self.db = None
        self.retrievals = None

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Set ingest params for evaluating pipeline.
        In this method, we don't ingest passages, because Qasper dataset is not designed for ingest all paragraphs
        and retrieve it. It only has questions that are related to certain papers.
        So, we ingest each paper's paragraphs when we evaluate it.
        :param retrievals: retrievals to ingest
        :param db: db to ingest
        :param ingest_size: Default is None. You don't need to set this params. If you set, it will ignore this param.
        """
        if ingest_size is not None:
            raise Warning("QasperEvaluator does not support ingest_size parameter. "
                          "You can adjust evaluate_size parameter in __init__ method.")
        self.db = db
        self.retrievals = retrievals

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on Qasper dataset.
        :return: EvaluateResult
        """
        result = None
        for idx, row in self.data.iterrows():
            self.__ingest_passages(row['passages'])
            evaluate_result = self._calculate_metrics(
                questions=row['question'],
                pipeline=self.run_pipeline,
                retrieval_gt=row['retrieval_gt'],
                answer_gt=[[answer] for answer in row['answer_gt']],
                **kwargs
            )
            if result is None:
                result = evaluate_result
            else:
                result += evaluate_result
            self.__delete_passages(row['passages'])

        return result

    def __ingest_passages(self, passages: List[Passage]):
        for retrieval in self.retrievals:
            retrieval.ingest(passages)
        self.db.create_or_load()
        self.db.save(passages)

    def __delete_passages(self, passages: List[Passage]):
        for retrieval in self.retrievals:
            retrieval.delete([passage.id for passage in passages])

    def preprocess(self, data):
        """
        Preprocess Qasper dataset to make it suitable for evaluating pipeline.
        """

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
