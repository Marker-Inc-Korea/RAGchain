import itertools
from copy import deepcopy
from typing import List, Optional

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseBeirEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult


class BeirNQEvaluator(BaseBeirEvaluator):
    """
    BeirNQEvaluator is a class for evaluating pipeline performance on NQ dataset at BEIR.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports KoStrategyQA.
        Supporting metrics are Recall, Precision, Hole, TopK_Accuracy, EM, F1_score, context_recall, context_precision, MRR.
        You must ingest all data for using context_recall and context_precision metrics.

        Additionally, you can preprocess datasets in this class constructor to benchmark your own pipeline.
        You can modify utils methods by overriding it for your dataset.
        """

        self.run_pipeline = run_pipeline
        self.eval_size = evaluate_size

        # Data load
        file_path = "BeIR/nq"

        # Create support metrics
        super().__init__(evaluate_size=self.eval_size, file_path=file_path, metrics=metrics)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If ingest size too big, It takes a long time.
        So we shuffle corpus and slice by ingest size for test.
        We put retrieval gt corpus in passages because retrieval retrieves ground truth in db.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """
        gt_ids = deepcopy(self.retrieval_gt)
        corpus = deepcopy(self.corpus)

        # Flatten retrieval ground truth ids and convert string type.
        gt_ids_lst = [str(id) for id in list(itertools.chain.from_iterable(gt_ids))]
        id_for_remove_duplicated_corpus = deepcopy(gt_ids_lst)

        # gt_passages is retrieval_gt passages to ingest.
        gt_passages = corpus.loc[corpus['_id'].isin(id_for_remove_duplicated_corpus)]

        # gt_passages = BaseBeirEvaluator.make_gt_passages(gt_ids, corpus)

        # Slice corpus by ingest_size and remove duplicate passages.
        corpus_passages = self.remove_duplicate_passages(ingest_size=ingest_size,
                                                         eval_size=self.eval_size,
                                                         corpus=corpus,
                                                         random_state=random_state,
                                                         id_for_remove_duplicated_corpus=id_for_remove_duplicated_corpus,
                                                         )

        gt_passages = gt_passages.apply(self.make_corpus_passages, axis=1).tolist()

        passages = corpus_passages.apply(self.make_corpus_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on nq dataset.
        This method always validate passages.
        """

        return self._calculate_metrics(
            questions=self.questions,
            pipeline=self.run_pipeline,
            retrieval_gt=self.retrieval_gt
        )
