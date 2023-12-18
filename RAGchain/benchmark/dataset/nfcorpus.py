from copy import deepcopy
from typing import List, Optional

import pandas as pd

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class NFCorpusEvaluator(BaseDatasetEvaluator):
    """
    NFCorpusEvaluator is a class for evaluating pipeline performance on NFCorpus dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        NFCorpus dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports NFCorpus dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
        'context_precision'.
        and rank aware metrics are 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR'.

        Notice:
        The reason context_recall does not accommodate this benchmark is due to the excessive number
        of retrieval ground truths that exceed the context length in ragas metrics.

        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """

        try:
            import ir_datasets
        except ImportError:
            raise ImportError('You have to pip install ir_datasets.\n '
                              'If it occurred error, please refer to our docs.')

        file_path = "nfcorpus/test"
        datasets = ir_datasets.load(file_path)

        query = pd.DataFrame({'query_id': query[0], 'query': query[1]} for query in datasets.queries_iter())
        doc = pd.DataFrame(
            {'doc_id': doc[0], 'doc': doc[3], 'title': doc[2], 'url': doc[1], 'doc_metadata': datasets.docs_metadata(),
                            'file_path': file_path} for doc in datasets.docs_iter())
        qrels = pd.DataFrame({'query_id': qrels[0], 'retrieval_gt': qrels[1], 'relevance': qrels[2]}
                             for qrels in datasets.qrels_iter() if qrels[2] > 1)

        qrels = qrels.groupby('query_id', as_index=False).agg(
            {'retrieval_gt': lambda x: list(x), 'relevance': lambda x: list(x)})

        default_metrics = self.retrieval_gt_metrics + self.retrieval_gt_metrics_rank_aware
        support_metrics = default_metrics + self.answer_no_gt_ragas_metrics

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

        if evaluate_size is not None and len(qrels) > evaluate_size:
            self.retrieval_gt = qrels[:evaluate_size]

        # Preprocess question, retrieval gt
        self.question = query[query['query_id'].isin(self.retrieval_gt['query_id'])]['query'].tolist()

        result = self.retrieval_gt.apply(self.__make_retrieval_gt, axis=1)
        self.gt, self.gt_ord = zip(*result)
        self.ingest_data = doc

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional

        Notice:
        If ingest size too big, It takes a long time.
        So we shuffle corpus and slice by ingest size for test.
        Put retrieval gt corpus in passages because retrieval retrieves ground truth in db.
        """
        ingest_data = deepcopy(self.ingest_data)
        id_for_remove_duplicated_docs = [gt for gt_lst in deepcopy(self.gt) for gt in gt_lst]

        # Create gt_passages for ingest.
        gt_passages = ingest_data[ingest_data['doc_id'].isin(id_for_remove_duplicated_docs)]
        gt_passages = gt_passages.apply(self.__make_passages, axis=1).tolist()

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data.sample(n=ingest_size, replace=False, random_state=random_state,
                                                 axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = ingest_data.isin(id_for_remove_duplicated_docs)
        # Remove duplicated passages
        ingest_data = ingest_data[~mask.any(axis=1)]
        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()

        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:

        return self._calculate_metrics(
            questions=self.question,
            pipeline=self.run_pipeline,
            retrieval_gt=list(self.gt),
            retrieval_gt_order=list(self.gt_ord),
            **kwargs
        )

    def __make_passages(self, row):

        return Passage(
            id=str(row['doc_id']),
            content=row['doc'],
            filepath=row['file_path'],
            metadata_etc={
                'title': row['title'],
                'url': row['url'],
                'count': row['doc_metadata']['count'],
                'fields': row['doc_metadata']['fields']
            }
        )

    def __make_retrieval_gt(self, row):
        gts = []
        gt_order = []
        for idx, gt in enumerate(row['retrieval_gt']):
            gts.append(gt)
            gt_order.append(row['relevance'][idx])

        return gts, gt_order
