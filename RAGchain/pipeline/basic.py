from operator import itemgetter
from typing import List, Optional, Union

from langchain.document_loaders.base import BaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from RAGchain.DB.base import BaseDB
from RAGchain.pipeline.base import BasePipeline, BaseRunPipeline
from RAGchain.preprocess.text_splitter import RecursiveTextSplitter
from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainPromptTemplate, RAGchainChatPromptTemplate
from RAGchain.utils.file_cache import FileCache


class BasicIngestPipeline(BasePipeline):
    """
    Basic ingest pipeline class.
    This class handles the ingestion process of documents into a database and retrieval system.
    First, load file from directory using file loader.
    Second, split document into passages using text splitter.
    Third, save passages to database.
    Fourth, ingest passages to retrieval module.

    :example:
    >>> from RAGchain.pipeline.basic import BasicIngestPipeline
    >>> from RAGchain.DB import PickleDB
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from RAGchain.preprocess.loader import FileLoader

    >>> file_loader = FileLoader(target_dir="./data")
    >>> db = PickleDB("./db")
    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> pipeline = BasicIngestPipeline(file_loader=file_loader, db=db, retrieval=retrieval)
    >>> pipeline.run()
    """

    def __init__(self,
                 file_loader: BaseLoader,
                 db: BaseDB,
                 retrieval: BaseRetrieval,
                 text_splitter: BaseTextSplitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50),
                 ignore_existed_file: bool = True):
        """
        Initialize BasicIngestPipeline.
        :param file_loader: File loader to load documents. You can use any file loader from langchain and RAGchain.
        :param db: Database to save passages.
        :param retrieval: Retrieval module to ingest passages.
        :param text_splitter: Text splitter to split document into passages. Default is RecursiveTextSplitter.
        :param ignore_existed_file: If True, ignore existed file in database. Default is True.
        """
        self.file_loader = file_loader
        self.text_splitter = text_splitter
        self.db = db
        self.retrieval = retrieval
        self.ignore_existed_file = ignore_existed_file

    def run(self, target_dir=None, *args, **kwargs):
        """
        Run ingest pipeline.

        :param target_dir: Target directory to load documents. If None, use target_dir from file_loader that you passed in __init__.
        """
        # File Loader
        if target_dir is not None:
            self.file_loader.target_dir = target_dir
        documents = self.file_loader.load()

        if self.ignore_existed_file:
            file_cache = FileCache(self.db)
            documents = file_cache.delete_duplicate(documents)

        if len(documents) <= 0:
            print("No file to ingest")
            return

        # Text Splitter
        passages = []
        for document in documents:
            passages.extend(self.text_splitter.split_document(document))
        print(f"Split into {len(passages)} passages")

        # Save passages to DB
        self.db.create_or_load()
        self.db.save(passages)

        # Ingest to retrieval
        self.retrieval.ingest(passages)
        print("Ingest complete!")


class BasicRunPipeline(BaseRunPipeline):
    """
    Basic run pipeline class.
    This class handles the run process of document question answering.
    First, retrieve passages from retrieval module.
    Second, run LLM module to get answer.
    Finally, you can get answer and passages as return value.

    :example:
    >>> from RAGchain.pipeline.basic import BasicRunPipeline
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from langchain.llms.openai import OpenAI

    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> pipeline = BasicRunPipeline(retrieval=retrieval, llm=OpenAI())
    >>> answer, passages, rel_scores = pipeline.get_passages_and_run(questions=["Where is the capital of Korea?"])
    >>> # Run with Langchain LECL
    >>> answer = pipeline.run.invoke({"question": "Where is the capital of Korea?"})
    """

    def __init__(self, retrieval: BaseRetrieval, llm: BaseLanguageModel,
                 prompt: Optional[Union[RAGchainPromptTemplate, RAGchainChatPromptTemplate]] = None,
                 retrieval_option: Optional[dict] = None):
        self.retrieval = retrieval
        self.llm = llm
        self.prompt = self._get_default_prompt(llm, prompt)
        self.retrieval_option = retrieval_option if retrieval_option is not None else {}
        super().__init__()

    def _make_runnable(self):
        self.run = RunnablePassthrough.assign(
            passages=itemgetter("question") | RunnableLambda(
                lambda question: Passage.make_prompts(self.retrieval.retrieve(question,
                                                                              **self.retrieval_option))),
            question=itemgetter("question"),
        ) | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        def passage_scores_to_dict(passages_scores):
            return {
                "passage_ids": passages_scores[0],
                "scores": passages_scores[1]
            }

        passage_scores = RunnablePassthrough.assign(
            passage_scores=itemgetter("question") | RunnableLambda(
                lambda question: passage_scores_to_dict(self.retrieval.retrieve_id_with_scores(
                    question, **self.retrieval_option)))
        )

        runnable = passage_scores | RunnablePassthrough.assign(
            passages=lambda x: self.retrieval.fetch_data(x['passage_scores']['passage_ids']),
            scores=lambda x: x['passage_scores']['scores']
        ) | RunnablePassthrough.assign(
            answer={
                       "question": itemgetter("question"),
                       "passages": itemgetter("passages") | RunnableLambda(
                           lambda passages: Passage.make_prompts(passages)
                       )
                   } | self.prompt | self.llm | StrOutputParser()
        )

        answers = runnable.batch(
            [{"question": question} for question in questions])

        final_answers, final_passages, final_scores = (
            map(list, zip(*[(answer['answer'], answer['passages'], answer['scores']) for answer in answers])))
        return final_answers, final_passages, final_scores
