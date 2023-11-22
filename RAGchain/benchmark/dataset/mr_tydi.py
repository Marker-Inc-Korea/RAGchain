from copy import deepcopy
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


# TODO: 진짜 mrr이 rank aware한것인가?
# TODO: positive passage는 여러개 있는가? 그리고 rank가 있는가> 근데 mrr이 있으면 rank가 있다고 추정
# TODO: train은 data leakage 일어날 수 있으니 dev로? -> dev는 303개 밖에 없는 데이터
# TODO: 문제점이 너무 한번 돌리는데 데이터셋 다운받는게 오래걸림
# TODO: 어떻게 language를 optional 하게 만들것인가?

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
        """
        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision', 'MRR']

        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.run_pipeline = run_pipeline

        # Data load
        self.file_path = 'castorini/mr-tydi'
        dataset = load_dataset(self.file_path, language)['train']
        corpus = load_dataset('castorini/mr-tydi-corpus', language)['train']
        self.corpus = pd.DataFrame(
            {'docid': corpus['docid'], 'title': corpus['title'], 'text': corpus['text']}
        )
        # TODO: ingest와 passage변환이 너무 오래걸려서 일시적으로 slice

        # Create qa data with train set for query, retrieval_gt
        # TODO: language가 잘못 들어갔을때의 경우의 수(여러 language를 받았을때-> 인제스트할때 다른 언어 데이터도 같이 인제스트 될 수 있으므로 그냥 사용자가 language알아서 바꿔가면서 돌리는게 좋을듯)
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
        """

        make_passages = deepcopy(self.corpus)
        passages = make_passages.apply(self.__make_corpus_passages, axis=1).tolist()
        test = passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)
        # TODO: 다른 language로 ingest할때는 그전건 초기화 시켜야함.

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
        passage = Passage(
            id=row['docid'],
            content=row['text'],
            filepath=self.file_path,
            metadata_etc={'title': row['title']}
        )
        return passage
