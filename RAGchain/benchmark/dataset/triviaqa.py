import itertools
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class TriviaQAEvaluator(BaseDatasetEvaluator):
    """
    TriviaQAEvaluator is a class for evaluating pipeline performance on TriviaQA dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        TriviaQA dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports TriviaQA dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
        'context_precision', 'BLEU', 'answer_relevancy', 'faithfulness', 'KF1',
        and rank aware metrics are 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR'.

        Notice:
        The reason context_recall does not accommodate this benchmark is due to the excessive number
        of retrieval ground truths that exceed the context length in ragas metrics.

        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """
        # TODO: Recommend make ingest size small in docs.
        #  This is because when ingesting data, having one query per ground truth becomes burdensome,
        #  especially when there are a large number of ground truths to ingest

        self.file_path = "trivia_qa"

        datasets = load_dataset("trivia_qa", "unfiltered")['validation'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.retrieval_gt_metrics_rank_aware \
                          + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_no_gt_ragas_metrics \
                          + self.answer_no_gt_ragas_metrics

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError("You input metrics that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size
        self.run_pipeline = run_pipeline

        # Preprocess dataset
        self.ingest_data = deepcopy(datasets[['question_id', 'question_source', 'search_results']])
        self.qa_data = datasets

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        """
        ingest_data = self.ingest_data
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Create passages.
        result = ingest_data.apply(self.__make_passages, axis=1)
        passages = list(itertools.chain.from_iterable(result))

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        result = self.qa_data.apply(self.__make_retrieval_gt, axis=1)
        gt, gt_order = zip(*result)
        answers = self.qa_data['answer'].apply(self.__make_answer_gt)

        return self._calculate_metrics(
            questions=self.qa_data['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=list(gt),
            retrieval_gt_order=list(gt_order),
            answer_gt=answers.tolist(),
            **kwargs
        )

    def __make_passages(self, row):
        passages = []
        search_results = row['search_results']

        for passage_idx, passage_text in enumerate(search_results['search_context']):
            passages.append(Passage(
                id=str(row['question_id']) + '_' + str(search_results['rank'][passage_idx]),
                content=passage_text,
                filepath=self.file_path,
                metadata_etc={
                    'description': search_results['description'][passage_idx],
                    'filename': search_results['filename'][passage_idx],
                    'title': search_results['title'][passage_idx],
                    'url': search_results['url'][passage_idx],
                    'rank': search_results['rank'][passage_idx]
                }
            ))
        return passages

    def __make_retrieval_gt(self, row):
        gt = []
        gt_order = []

        search_results = row['search_results']
        for idx, rank in enumerate(search_results['rank']):
            gt.append(str(row['question_id']) + '_' + str(search_results['rank'][idx]))
            gt_order.append(50 - (rank))

        return gt, gt_order

    def __make_answer_gt(self, row):

        return [answer for answer in row['normalized_aliases']]
