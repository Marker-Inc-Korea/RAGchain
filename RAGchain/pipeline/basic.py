from operator import itemgetter
from typing import List, Optional

import langchain
from langchain.document_loaders.base import BaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from RAGchain.DB.base import BaseDB
from RAGchain.pipeline.base import BasePipeline, BaseRunPipeline
from RAGchain.preprocess.text_splitter import RecursiveTextSplitter
from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainPromptTemplate
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
    default_prompt = RAGchainPromptTemplate.from_template(
        """
        Given the information, answer the question. If you don't know the answer, don't make up 
        the answer, just say you don't know.
        
        Information :
        {passages}
        
        Question: {question}
        
        Answer:
        """
    )

    def __init__(self, retrieval: BaseRetrieval, llm: langchain.llms.base.BaseLLM,
                 prompt: Optional[RAGchainPromptTemplate] = None,
                 retrieval_option: Optional[dict] = None):
        self.retrieval = retrieval
        self.llm = llm
        self.prompt = prompt if prompt is not None else self.default_prompt
        self.retrieval_option = retrieval_option if retrieval_option is not None else {}
        super().__init__()

    def _make_runnable(self):
        self.run = {
                       "passages": itemgetter("question") | RunnableLambda(
                           lambda question: self.retrieval.retrieve(question, **self.retrieval_option)),
                       "question": itemgetter("question"),
                   } | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        passage_ids, scores = map(list,
                                  zip(*[self.retrieval.retrieve_id_with_scores(question, **self.retrieval_option)
                                        for question in questions]))
        passages = list(map(self.retrieval.fetch_data, passage_ids))

        runnable = {
                       "question": itemgetter("question"),
                       "passages": itemgetter("passages") | RunnableLambda(lambda x: Passage.make_prompts(x))
                   } | self.prompt | self.llm | StrOutputParser()
        answers = runnable.batch(
            [{"question": question, "passages": passage_group} for question, passage_group in zip(questions, passages)])
        return answers, passages, scores
