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

        Notice:
        The default metric refers to the metric that is essentially executed when you run the test file.
        Support metrics refer to those that are available for use.
        This distinction exists because the evaluation process for Ragas metrics is time-consuming.
        """
        default_metrics = self.retrieval_gt_metrics + ['MRR']
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics + self.retrieval_no_gt_ragas_metrics \
                          + self.answer_no_gt_ragas_metrics
        languages = ['arabic', 'bengali', 'combined', 'english', 'finnish',
                     'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']
        language = language.lower()

        if language not in languages:
            raise ValueError(f"You input invalid language ({language})."
                             f"\nPlease input language among below language."
                             "\n(arabic, bengali, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai)")

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError(f"You input {metric} that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

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
        You must ingest all data for using context_recall metrics.
        If the ingest size is excessively large, it results in prolonged processing times.
        To address this, we shuffle the corpus and slice it according to the ingest size for testing purposes.
        The reason for transforming the retrieval ground truth corpus into passages and ingesting it is to enable
        retrieval to retrieve the retrieval ground truth within the database.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """

        gt_ids = deepcopy(self.qa_data['positive_passages'])
        corpus_passages = deepcopy(self.corpus)

        gt_ids = gt_ids.apply(self.__extract_gt_id)
        gt_ingestion = list(itertools.chain.from_iterable(gt_ids))

        # Setting the evaluation size.
        if self.eval_size is None:
            eval_size = len(gt_ingestion)
        else:
            eval_size = self.eval_size

        self.__validate_eval_size_and_ingest_size(ingest_size, eval_size)

        # Create gt_passages for ingest.
        gt_passages = corpus_passages[corpus_passages['docid'].isin(gt_ingestion)]
        gt_passages = gt_passages.apply(self.__make_corpus_passages, axis=1).tolist()

        if ingest_size is not None:
            corpus_passages = corpus_passages.sample(n=ingest_size, replace=False, random_state=random_state,
                                                     axis=0)

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = corpus_passages.isin(gt_ingestion)
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

    def __validate_eval_size_and_ingest_size(self, ingest_size, eval_size):
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size < eval_size:
                raise ValueError(f"ingest size({ingest_size}) must be same or larger than evaluate size({eval_size})")
