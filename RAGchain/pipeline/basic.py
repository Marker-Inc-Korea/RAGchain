from typing import List

from dotenv import load_dotenv
from langchain.document_loaders.base import BaseLoader

from RAGchain.DB.base import BaseDB
from RAGchain.llm.base import BaseLLM
from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline.base import BasePipeline
from RAGchain.preprocess.text_splitter import RecursiveTextSplitter
from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.file_cache import FileCache
from RAGchain.utils.util import slice_stop_words


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
        load_dotenv(verbose=False)

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


class BasicDatasetPipeline(BasePipeline):
    """
    DEPRECATED
    This class is deprecated and recommend to use BasicIngestPipeline instead.
    Basic dataset pipeline class.
    You can ingest specific dataset to retrieval system.
    """
    def __init__(self, file_loader: BaseLoader, retrieval: BaseRetrieval):
        self.file_loader = file_loader
        self.retrieval = retrieval
        load_dotenv(verbose=False)

    def run(self, *args, **kwargs):
        # File Loader
        documents = self.file_loader.load()
        if len(documents) <= 0:
            return

        passages = [Passage(id=document.metadata['id'], content=document.page_content,
                            filepath='KoStrategyQA', previous_passage_id=None, next_passage_id=None) for document in
                    documents]
        # Ingest to retrieval
        self.retrieval.ingest(passages)


class BasicRunPipeline(BasePipeline):
    """
    Basic run pipeline class.
    This class handles the run process of document question answering.
    First, retrieve passages from retrieval module.
    Second, run LLM module to get answer.
    Finally, you can get answer and passages as return value.

    :example:
    >>> from RAGchain.pipeline.basic import BasicRunPipeline
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from RAGchain.llm.basic import BasicLLM

    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> llm = BasicLLM(retrieval)
    >>> pipeline = BasicRunPipeline(retrieval=retrieval, llm=llm)
    >>> answer, passages = pipeline.run(query="Where is the capital of Korea?")
    """
    def __init__(self,
                 retrieval: BaseRetrieval,
                 llm: BaseLLM = None):
        """
        Initialize BasicRunPipeline.
        :param retrieval: Retrieval module to retrieve passages.
        :param llm: LLM module to get answer. Default is BasicLLM.
        """
        load_dotenv()
        self.retrieval = retrieval
        self.llm = llm if llm is not None else BasicLLM(retrieval)

    def run(self, query: str, *args, **kwargs) -> tuple[str, List[Passage]]:
        """
        Run the run pipeline.
        :param query: Query to ask.
        :return: Answer, retrieved passages.
        """
        answer, passages = self.llm.ask(query=query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        return answer, passages
