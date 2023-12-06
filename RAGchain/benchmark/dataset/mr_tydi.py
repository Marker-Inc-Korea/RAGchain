import itertools
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class MrTydiEvaluator(BaseDatasetEvaluator):
    """
    MrTydiEvaluator is a class for evaluating pipeline performance on Mr.tydi dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 language: str = 'english'):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports KoStrategyQA.
        Supporting metrics are Recall, Precision, Hole, TopK_Accuracy, EM, F1_score, context_recall, context_precision, MRR.
        You must ingest all data for using context_recall and context_precision metrics.
        :param language: The string data which is Mr.tydi dataset language. Default is english.
        You can choose languages like below.
        arabic, bengali, combined, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai
        If you want to use languages combined, You can choose 'combined' configuration.
        """
        support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics + self.retrieval_gt_ragas_metrics
                           ['MRR'])
        languages = ['arabic', 'bengali', 'combined', 'english', 'finnish',
                     'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']

        if language not in languages:
            raise ValueError(f"You input invalid language ({language})."
                             f"\nPlease input language among below language."
                             "\n(arabic, bengali, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai)")

        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.run_pipeline = run_pipeline
        self.eval_size = evaluate_size

        # Data load
        self.file_path = 'castorini/mr-tydi'
        dataset = load_dataset(self.file_path, language)['test']
        corpus = load_dataset('castorini/mr-tydi-corpus', language)['train']

        # Convert dataformat as pandas dataframe
        self.qa_data = dataset.to_pandas()
        self.corpus = corpus.to_pandas()

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If ingest size too big, It takes a long time.
        So we shuffle corpus and slice by ingest size for test.
        Put retrieval gt corpus in passages because retrieval retrieves ground truth in db.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """

        gt_ids = deepcopy(self.qa_data['positive_passages'])
        corpus_passages = deepcopy(self.corpus)

        gt_ids = gt_ids.apply(self.__extract_gt_id)
        id_for_remove_duplicated_corpus = list(itertools.chain.from_iterable(gt_ids))

        # Create gt_passages for ingest.
        gt_passages = corpus_passages[corpus_passages['docid'].isin(id_for_remove_duplicated_corpus)]
        gt_passages = gt_passages.apply(self.__make_corpus_passages, axis=1).tolist()

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                corpus_passages = corpus_passages.sample(n=ingest_size, replace=False, random_state=random_state,
                                                         axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = corpus_passages.isin(id_for_remove_duplicated_corpus)
        # Remove duplicated passages
        corpus_passages = corpus_passages[~mask.any(axis=1)]
        passages = corpus_passages.apply(self.__make_corpus_passages, axis=1).tolist()

        passages += gt_passages
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on Mr. Tydi dataset.
        This method always validate passages.
        """
        retrieval_gt = [[passage['docid'] for passage in passages] for passages in
                        self.qa_data['positive_passages'].tolist()]

        return self._calculate_metrics(
            questions=self.qa_data['query'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt
        )

    def __make_corpus_passages(self, row):
        # Corpus to passages
        passage = Passage(
            id=row['docid'],
            content=row['text'],
            filepath=self.file_path,
            metadata_etc={'title': row['title']}
        )
        return passage

    def __extract_gt_id(self, row):
        # retrieval gt to passage for ingest.
        # gt_id for removing duplicated value
        gt_id = []
        for element in row:
            gt_id.append(element['docid'])
        return gt_id
