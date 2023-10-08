from typing import List, Dict

from dotenv import load_dotenv
from langchain.document_loaders.base import BaseLoader

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.pipeline.base import BasePipeline
from KoPrivateGPT.preprocess.text_splitter import RecursiveTextSplitter
from KoPrivateGPT.preprocess.text_splitter.base import BaseTextSplitter
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.file_cache import FileCache
from KoPrivateGPT.utils.util import slice_stop_words


class BasicIngestPipeline(BasePipeline):
    """
    Class representing a basic ingest pipeline.

    This class handles the ingestion process of documents into a database and retrieval system.

    Attributes:
        file_loader (langchain.document_loaders.base.BaseLoader): The file loader instance want to use.
        text_splitter (tuple[str, Dict[str, Any]]): The type and configuration of the text splitter module.
        db (tuple[str, Dict[str, Any]]): The type and configuration of the database module.
        retrieval (tuple[str, Dict[str, Any]]): The type and configuration of the retrieval module.
        ignore_existed_file (bool): A flag indicating whether to ignore already ingested files.

    Methods:
        run: Runs the pipeline to ingest documents.

    """

    def __init__(self,
                 file_loader: BaseLoader,
                 db: BaseDB,
                 retrieval: BaseRetrieval,
                 text_splitter: BaseTextSplitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50),
                 ignore_existed_file: bool = True):
        self.file_loader = file_loader
        self.text_splitter = text_splitter
        self.db = db
        self.retrieval = retrieval
        self.ignore_existed_file = ignore_existed_file
        load_dotenv(verbose=False)

    def run(self, target_dir=None, *args, **kwargs):
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
    def __init__(self,
                 retrieval: BaseRetrieval,
                 llm: BaseLLM = None):
        load_dotenv()
        self.retrieval = retrieval
        self.llm = llm if llm is not None else BasicLLM(retrieval)

    def run(self, query: str, *args, **kwargs) -> tuple[str, List[Passage]]:
        answer, passages = self.llm.ask(query=query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        return answer, passages
