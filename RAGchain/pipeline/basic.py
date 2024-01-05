from typing import List, Optional, Union

from langchain.document_loaders.base import BaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableLambda

from RAGchain.DB.base import BaseDB
from RAGchain.pipeline.base import BaseIngestPipeline, BaseRunPipeline
from RAGchain.preprocess.text_splitter import RecursiveTextSplitter
from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainPromptTemplate, RAGchainChatPromptTemplate, RetrievalResult


class BasicIngestPipeline(BaseIngestPipeline):
    """
    Basic ingest pipeline class.
    This class handles the ingestion process of documents into a database and retrieval system.
    First, load file from directory using file loader.
    Second, split a document into passages using text splitter.
    Third, save passages to a database.
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
    >>> pipeline.run.invoke(None)
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
        :param text_splitter: Text splitter to split a document into passages. Default is RecursiveTextSplitter.
        :param ignore_existed_file: If True, ignore existed file in a database. Default is True.
        """
        self.file_loader = file_loader
        self.text_splitter = text_splitter
        self.db = db
        self.retrieval = retrieval
        self.ignore_existed_file = ignore_existed_file
        super().__init__()

    def _make_runnable(self):
        self.run = (RunnableLambda(self.file_loader.load) |
                    self.text_splitter |
                    self.db | self.retrieval.as_ingest())


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
    >>> # Run with Langchain LCEL
    >>> answer = pipeline.run.invoke("Where is the capital of Korea?")
    """

    def __init__(self, retrieval: BaseRetrieval, llm: BaseLanguageModel,
                 prompt: Optional[Union[RAGchainPromptTemplate, RAGchainChatPromptTemplate]] = None):
        self.retrieval = retrieval
        self.llm = llm
        self.prompt = self._get_default_prompt(llm, prompt)
        super().__init__()

    def _make_runnable(self):
        self.run = self.retrieval | RunnableLambda(
            RetrievalResult.to_prompt_input) | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str], top_k: int = 5) -> tuple[
        List[str], List[List[Passage]], List[List[float]]]:
        runnable = self.retrieval | {
            "answer": RunnableLambda(lambda x: x.to_prompt_input()) | self.prompt | self.llm | StrOutputParser(),
            "passages": RunnableLambda(lambda x: x.passages),
            "scores": RunnableLambda(lambda x: x.scores),
        }
        answers = runnable.batch(questions, config={"configurable": {"retrieval_options": {"top_k": top_k}}})

        final_answers, final_passages, final_scores = (
            map(list, zip(*[(answer['answer'], answer['passages'], answer['scores']) for answer in answers])))
        return final_answers, final_passages, final_scores
