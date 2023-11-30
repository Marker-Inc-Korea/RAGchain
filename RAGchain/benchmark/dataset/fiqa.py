import itertools
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseBeirEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class BeirFIQAEvaluator(BaseBeirEvaluator):
    """
    BeirFIQAEvaluator is a class for evaluating pipeline performance on FIQA dataset at BEIR.
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
        support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics)
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.run_pipeline = run_pipeline
        self.eval_size = evaluate_size

        # Data load
        self.file_path = "BeIR/fiqa"
        queries = load_dataset(self.file_path, 'queries')['queries']
        corpus = load_dataset(self.file_path, 'corpus')['corpus']
        qrels = load_dataset(f"{self.file_path}-qrels")['test']

        self.queries = queries.to_pandas()
        self.corpus = corpus.to_pandas()
        self.qrels = qrels.to_pandas()

        # Convert integer type to string type of qrels' query-id and corpus-id.
        self.qrels[['query-id', 'corpus-id']] = self.qrels[['query-id', 'corpus-id']].astype(str)

        # Preprocess qrels. Some query ids duplicated and were appended different corpus id.
        self.qrels = self.qrels.groupby('query-id', as_index=False).agg(
            {'corpus-id': lambda x: list(x), 'score': lambda x: list(x)})

        if evaluate_size is not None and len(self.qrels) > evaluate_size:
            self.qrels = self.qrels[:evaluate_size]

        # Create retrieval_gt.
        self.retrieval_gt = self.qrels['corpus-id'].tolist()

        # Create question
        q_id = self.qrels['query-id'].tolist()
        self.questions = self.queries.loc[self.queries['_id'].isin(q_id)]['text'].tolist()

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

        # Slice corpus by ingest_size and remove duplicate passages.
        corpus_passages = self.remove_duplicate_passages(ingest_size=ingest_size,
                                                         eval_size=self.eval_size,
                                                         corpus=corpus,
                                                         random_state=random_state,
                                                         id_for_remove_duplicated_corpus=id_for_remove_duplicated_corpus,
                                                         )

        gt_passages = gt_passages.apply(self.__make_corpus_passages, axis=1).tolist()

        passages = corpus_passages.apply(self.__make_corpus_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on fiqa dataset.
        This method always validate passages.
        """
        return self._calculate_metrics(
            questions=self.questions,
            pipeline=self.run_pipeline,
            retrieval_gt=self.retrieval_gt
        )

    def __make_corpus_passages(self, row):
        # Corpus to passages
        passage = Passage(
            id=row['_id'],
            content=row['text'],
            filepath=self.file_path,
            metadata_etc={'title': row['title']}
        )
        return passage

    # TODO: qrels 아이디랑 코퍼스 쿼리 아이디 매핑하는 함수 만들기 함수화 시키기
