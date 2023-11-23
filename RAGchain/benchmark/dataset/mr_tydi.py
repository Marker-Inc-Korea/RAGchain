import itertools
from copy import deepcopy
from typing import List, Optional

import pandas as pd
import sklearn
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class MrTydiEvaluator(BaseDatasetEvaluator):
    """
    MrTydiEvaluator is a class for evaluating pipeline performance on StrategyQA dataset.
    """

    def __init__(self, run_pipeline: BasePipeline,
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
        :param language: The string data which is Mr.tydi dataset language.
        You can choose languages like below.
        arabic, bengali, combined, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai
        """
        support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics +
                           ['MRR'])
        languages = ['arabic', 'bengali', 'combined', 'english', 'finnish',
                     'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']

        if language not in languages:
            raise ValueError(f"You input invalid language ({language})."
                             f"\nPlease input language among below language."
                             "\n(arabic, bengali, combined, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai)")

        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.run_pipeline = run_pipeline
        self.eval_size = evaluate_size

        # Data load
        self.file_path = 'castorini/mr-tydi'
        dataset = load_dataset(self.file_path, language)['train']
        corpus = load_dataset('castorini/mr-tydi-corpus', language)['train']
        self.corpus = pd.DataFrame(
            {'docid': corpus['docid'], 'title': corpus['title'], 'text': corpus['text']}
        )

        # Create qa data with train set for query, retrieval_gt
        self.qa_data = pd.DataFrame(
            {'query_id': dataset['query_id'], 'question': dataset['query'],
             'positive_passages': dataset['positive_passages']}
        )
        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """

        # If ingest size too bit, It takes a long time.(Because corpus size is 1496126!)
        # So we shuffle corpus and slice by ingest size for test.
        # Put retrieval gt corpus in passages because retrieval retrieves ground truth in db.

        make_retrieval_gt_passage = deepcopy(self.qa_data['positive_passages'])
        make_passages = deepcopy(self.corpus)

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                make_passages = sklearn.utils.shuffle(make_passages)[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        make_retrieval_gt_passage = make_retrieval_gt_passage.apply(self.__make_corpus_passages)
        make_retrieval_gt_passage, gt_ids = zip(*make_retrieval_gt_passage)
        id_for_remove_duplicated_corpus = list(itertools.chain.from_iterable(gt_ids))

        # Marking dupicated values in the corpus using retrieval_gt id.
        mask = make_passages.isin(id_for_remove_duplicated_corpus)
        # Remove duplicated passages
        make_passages = make_passages[~mask.any(axis=1)]

        make_retrieval_gt_passage = list(itertools.chain.from_iterable(make_retrieval_gt_passage))
        passages = make_passages.apply(self.__make_corpus_passages, axis=1).tolist()
        for gt_passage in make_retrieval_gt_passage:
            passages.append(gt_passage)

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)
        # TODO: 다른 language로 ingest할때는 그전건 초기화 시켜야함.
        # TODO: 공식문서 쓸때 내 velog link 걸기(영어로 번역)

    def evaluate(self, **kwargs) -> EvaluateResult:
        retrieval_gt = [[passage_id['docid'] for passage_id in passages] for passages in
                        self.qa_data['positive_passages'].tolist()]

        return self._calculate_metrics(
            questions=self.qa_data['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            **kwargs
        )

    def __make_corpus_passages(self, row):
        # retrieval gt to passage for ingest.
        if type(row) == list:
            gt_passages = []
            # gt_id for removing duplicated value
            gt_id = []
            for element in row:
                gt_passages.append(Passage(
                    id=element['docid'],
                    content=element['text'],
                    filepath=self.file_path,
                    metadata_etc={'title': element['title']}
                ))

                gt_id.append(element['docid'])
            return gt_passages, gt_id

        # Corpus to passages
        else:
            passage = Passage(
                id=row['docid'],
                content=row['text'],
                filepath=self.file_path,
                metadata_etc={'title': row['title']}
            )
            return passage
