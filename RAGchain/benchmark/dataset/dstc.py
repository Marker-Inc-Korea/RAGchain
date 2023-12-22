import itertools
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class DSTCEvaluator(BaseDatasetEvaluator):
    """
    DSTCEvaluator is a class for evaluating pipeline performance on DSTC-11-Track-5 dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        DSTC-11-Track-5 dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports DSTC-11-Track-5 dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
        'context_precision', 'answer_relevancy', 'faithfulness'.
        Rank aware metrics are 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR'.
        You must ingest all data for using context_recall and context_precision metrics.

        Notice:
        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """

        # TODO: 앞써 한 발화는 따로 문맥같은걸로 question에 포함해야하는가?

        self.file_path = "NomaDamas/DSTC-11-Track-5"
        qa = load_dataset(self.file_path, 'default')['test'].to_pandas().dropna()
        self.knowledge = load_dataset(self.file_path, 'knowledge')['train'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics \
                          + self.retrieval_no_gt_ragas_metrics + self.answer_no_gt_ragas_metrics

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError(f"You input {metric} that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size
        self.run_pipeline = run_pipeline

        # Preprocess qa and knowledge data
        self.questions, self.retrieval_gt, self.response = zip(*qa.apply(self.__preprocess_qa, axis=1))
        self.knowledge['doc_id'] = self.knowledge.apply(self.__renewal_doc_id, axis=1)

        test = self.retrieval_gt

        if evaluate_size is not None and len(qa) > evaluate_size:
            self.questions = self.questions[:evaluate_size]
            self.retrieval_gt = self.retrieval_gt[:evaluate_size]
            self.response = self.response[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        You must ingest all data for using context_recall metrics.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """
        ingest_data = deepcopy(self.knowledge)
        gt_ids = deepcopy(self.retrieval_gt)

        id_for_remove_duplicated_corpus = list(itertools.chain.from_iterable(gt_ids))

        # Create gt_passages for ingest.
        gt_passages = ingest_data[ingest_data['doc_id'].isin(id_for_remove_duplicated_corpus)]
        gt_passages = gt_passages.apply(self.__make_passages, axis=1).tolist()
        # TODO: 자꾸 주석실수하니 metric 관련 주석 부분은 rank aware한것, answer있는것, gt만 있는것 따로해서 모듈화 시켜 상속

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data.sample(n=ingest_size, replace=False, random_state=random_state,
                                                 axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = ingest_data.isin(id_for_remove_duplicated_corpus)
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
            questions=list(self.questions),
            pipeline=self.run_pipeline,
            retrieval_gt=list(self.retrieval_gt),
            answer_gt=list([answer] for answer in self.response),
            **kwargs
        )

    def __make_passages(self, row):
        if row['doc_type'] == 'review':
            content = row['review_sentence']
            metadata_etc = {
                'domain': row['domain'],
                'entity_id': row['entity_id'],
                'entity_name': row['entity_name'],
                'doc_type': row['doc_type'],
                **row['review_metadata']
            }

        elif row['doc_type'] == 'faq':
            content = row['faq_question'] + ', ' + row['faq_answer']
            metadata_etc = {
                'domain': row['domain'],
                'entity_id': row['entity_id'],
                'entity_name': row['entity_name'],
                'doc_type': row['doc_type'],
            }

        return Passage(
            id=row['doc_id'],
            content=content,
            filepath=self.file_path,
            metadata_etc=metadata_etc
        )

    def __renewal_doc_id(self, row):
        if row['doc_type'] == 'review':
            return str(row['doc_id']) + '_' + row['doc_type'] + '_' + row['domain'] \
                + '_' + str(row['entity_id']) + '_' + str(row['review_sent_id'])
        elif row['doc_type'] == 'faq':
            return str(row['doc_id']) + '_' + row['doc_type'] + '_' \
                + row['domain'] + '_' + str(row['entity_id'])

    def __preprocess_qa(self, row):
        question = row['log'][-1]['text']
        response = row['response']
        gt = []
        for knowledge in row['knowledge']:
            if knowledge['doc_type'] == 'review':
                gt.append(str(knowledge['doc_id']) + '_' + knowledge['doc_type'] + '_' + knowledge['domain'] + '_'
                          + str(knowledge['entity_id']) + '_' + str(int(knowledge['sent_id'])))
            elif knowledge['doc_type'] == 'faq':
                gt.append(str(knowledge['doc_id']) + '_' + knowledge['doc_type'] + '_'
                          + knowledge['domain'] + '_' + str(knowledge['entity_id']))

        return question, gt, response

# TODO: 모두 대화의 마지막이 user로 끝나는가?
