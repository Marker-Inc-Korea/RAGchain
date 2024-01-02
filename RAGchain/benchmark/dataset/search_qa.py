import itertools
import uuid
from copy import deepcopy
from typing import List, Optional

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class SearchQAEvaluator(BaseDatasetEvaluator):
    """
    SearchQAEvaluator is a class for evaluating pipeline performance on search qa dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        natural qa dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports natural qa dataset.
        Supporting metrics are 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'Recall', 'Precision'
        'context_recall', 'context_precision', 'BLEU', 'answer_relevancy', 'faithfulness', 'KF1'.
        You must ingest all data for using context_recall metrics.

        Notice:
        Default metrics are essentially the metrics run when executing a test file.
        Support metrics refer to the available metrics.
        This distinction arises due to the prolonged evaluation time required for Ragas metrics.
        """

        import pandas as pd
        from datasets import load_dataset, Dataset

        self.file_path = "search_qa"
        dataset = load_dataset(self.file_path, 'raw_jeopardy')['train'].to_pandas()
        dataset['value'] = dataset['value'].fillna('$0')
        dataset = dataset.dropna()
        metadata = dataset[['air_date', 'category', 'value', 'round', 'show_number']]

        dataset = dataset.drop(['air_date', 'category', 'value', 'round', 'show_number'], axis=1)

        # dataset에 query_id 추가
        dataset = pd.Series([str(uuid.uuid4()) for _ in range(len(dataset))], name='query_id').to_frame().join(dataset)

        # preprocessing corpus
        # task: retrieval gt를 만들기 위한 corpus(metadata가 없음)와 ingestion을 위한 corpus(metadata가 있음)를 만들어야함
        corpus = dataset['search_results'].apply(pd.Series).drop(['related_links'], axis=1)

        # TODO: query하나당 똑같은 snippet이 나오지 체크
        for i in corpus['snippets']:
            test = set(i)
            if len(test) != len(i):
                raise ValueError("There is same snippets in corpus['snippets']")

        corpus = corpus.drop(['titles', 'urls'], axis=1)

        # corpus를 metadata와 분리 어차피 그룹바이하고나서 meta_data는 붙여바로면 됌

        # corpus는 query_id/snippet/air_date/category/value/round/show_number형태임
        corpus = pd.concat([dataset['query_id'], corpus, metadata], axis=1)  # metadata 있는 corpus용 corpus

        # TODO: test코드이므로 지우기
        check_corpus_isnull = corpus.isnull().sum()  # 모두 null값 없음
        # assert check_corpus_isnull['snippets'] == 0  # corpus에 null값이 있는지 확인(특히 snippets에 null값이 있는지 확인)

        for_checking_lost_snippets = 0
        for i in corpus['snippets']:
            for_checking_lost_snippets += len(i)
            # if None in i:
            #     raise ValueError("There is None in corpus['snippets']")
            # TODO: 1월/2일 여기서 부터 snippet 자체에서 null값이 있는것을 확인했음 -> 그렇다면 모두 null이라 query id 자체가 날라가는 것도 있는가?
            # TODO: 이것을 확인하기 위해서 corpus의 query_id와 원본 query_id를 비교해보자 실험

        corpus = corpus.explode(['snippets']).reset_index(drop=True)

        # Check if snippet are lost or not.
        assert len(corpus) == for_checking_lost_snippets

        check_explod_corpus_isnull = corpus.isnull().sum()  # corpus에 null값이 있는지 확인(특히 snippets에 null값이 있는지 확인)
        # TODO: explode가 문제인것을 확인 -> 그렇다면 원래 snippets이 null값이었던것인가?
        # assert check_explod_corpus_isnull['snippets'] == 0  # corpus에 null값이 있는지 확인(특히 snippets에 null값이 있는지 확인)

        # TODO: dropna가 문제인지 체크하기 위해서 따로 빼놓음
        corpus = corpus.dropna()

        gt_corpus = corpus.drop(['air_date', 'category', 'value', 'round', 'show_number'], axis=1)  # gt를 만들기 위한 corpus

        copy_corpus = deepcopy(corpus)
        copy_corpus = copy_corpus.drop_duplicates(subset='snippets', keep='first')

        # TODO: 여기서 snippet 외의 none값이 발생되어서 원래 있어야할 snippet과 doc_id가 사라져 매칭이 안되는 억까 ->
        check_isnull_after_drop_duplicates = copy_corpus.isnull().sum()  # 중복된 snippet은 제거된 상태, 즉 순수한 corpus 추출

        copy_corpus = copy_corpus.dropna()

        # 중복된 snippet은 제거된 상태, 즉 순수한 corpus 추출, doc_id생성
        doc_id = copy_corpus.apply(self.__make_doc_id, axis=1)
        doc_id = doc_id.rename('doc_id', inplace=True)
        copy_corpus = pd.concat([copy_corpus, doc_id], axis=1)
        complete_corpus = copy_corpus  # corpus로 hugging에 push할것(query_id/snippet/air_date/category/value/round/show_number/doc_id)

        # 순수한 copy_corpus를 가지고 gt를 만들기 위해 원본 corpus를 이용해서 query_id를 매칭시켜 gt를 부여
        gt_copy_corpus = copy_corpus.drop(['air_date', 'category', 'value', 'round', 'show_number'], axis=1)
        merged_df = gt_corpus.merge(gt_copy_corpus, on='snippets', how='left').drop(['query_id_y'], axis=1).rename(
            columns={'query_id_x': 'query_id'})

        # 여기서 null 값이 있다면 snippets에 null값이 있었다는 뜻임 -> retrieval gt fetch할때 문제
        check_isnull_merged_df = merged_df.isnull().sum()

        # merge한 다음 groupby해서 corpus 만들기 -> groupby했을때 원본데이터의 shape이랑 같아야함.
        # 원본 qa_data와의 shape를 같게 맞춰주기 위해서 query_id를 groupby해서 retrieval_gt리스트들을 만들어줌
        create_gt = (merged_df.groupby('query_id', as_index=False).agg({'snippets': lambda x: list(x),
                                                                        'doc_id': lambda x: list(x)}))
        # Remove none gt rows because of corpus can't retrieve none gt and removed query in preprocessing process.
        mask = dataset['query_id'].isin(create_gt['query_id'])
        dataset = dataset[mask]
        assert len(create_gt) == len(
            dataset)  # 원본 qa_data와의 shape를 같게 맞춰주기 위해서 query_id를 groupby해서 retrieval_gt리스트들을 만들어줌

        # qa_data에 doc_id(gt) 추가
        from sklearn.model_selection import train_test_split
        qa_dataset = pd.concat([dataset, create_gt['doc_id']], axis=1).rename(columns={'doc_id': 'retrieval_gt'})

        check_qadata_isnull = qa_dataset.isnull().sum()  # qa_dataset에 null값이 있는지 확인

        qa_dataset_train, qa_dataset_test = train_test_split(qa_dataset, test_size=0.2, shuffle=False)

        qa_dataset_train = Dataset.from_pandas(qa_dataset_train)
        qa_dataset_test = Dataset.from_pandas(qa_dataset_test)
        complete_corpus = Dataset.from_pandas(complete_corpus)

        qa_dataset_train.push_to_hub("NomaDamas/searchqa-split", 'qa_data', split='train')
        qa_dataset_test.push_to_hub("NomaDamas/searchqa-split", 'qa_data', split='test')
        complete_corpus.push_to_hub("NomaDamas/searchqa-split", 'corpus')

        # ------------------------------------------------------------

        # TODO: pandas seriese 고급파이썬 공부

        self.file_path = "NomaDamas/searchqa-split"
        self.qa_data = load_dataset(self.file_path, 'qa_data')['test'].to_pandas()
        self.corpus = load_dataset(self.file_path, 'corpus')['train'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics + self.retrieval_no_gt_ragas_metrics \
                          + self.answer_no_gt_ragas_metrics

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

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If the ingest size is excessively large, it results in prolonged processing times.
        To address this, we shuffle the corpus and slice it according to the ingest size for testing purposes.
        The reason for transforming the retrieval ground truth corpus into passages and ingesting it is to enable
        retrieval to retrieve the retrieval ground truth within the database.
        This dataset has many retrieval ground truths per query, so it is recommended to set the ingest size to a small value.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """

        ingest_data = deepcopy(self.corpus)
        gt_ingestion = list(itertools.chain.from_iterable(deepcopy([gt for gt in self.qa_data['retrieval_gt']])))

        # Retrieval ground truth ingestion -> 여기서 잘못된건가?
        gt_df = ingest_data[ingest_data['doc_id'].isin(gt_ingestion)]
        gt_passages = gt_df.apply(self.__make_passages, axis=1).tolist()

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data.sample(n=ingest_size, replace=False, random_state=random_state,
                                                 axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = ingest_data.isin(gt_ingestion)
        # Remove duplicated passages
        ingest_data = ingest_data[~mask.any(axis=1)]

        # Create passages.
        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.qa_data['question'].tolist()
        retrieval_gt = [[uuid.UUID(gt) for gt in gt_lst] for gt_lst in self.qa_data['retrieval_gt']]
        answer_gt = [answer for answer in self.qa_data['answer']]

        return self._calculate_metrics(
            questions=question,
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answer_gt,
            **kwargs
        )

    def __make_passages(self, row):

        return Passage(
            id=row['doc_id'],
            content=row['snippets'],
                filepath=self.file_path,
                metadata_etc={
                    'air_date': row['air_date'],
                    'category': row['category'],
                    'value': row['value'],
                    'round': row['round'],
                    'show_number': row['show_number']
                }
        )

    def __make_doc_id(self, row):

        return str(uuid.uuid4())
