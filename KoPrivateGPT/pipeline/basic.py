from typing import List
from uuid import UUID

from dotenv import load_dotenv

from KoPrivateGPT.options import Options, PickleDBOptions
from KoPrivateGPT.options.config import MongoDBOptions
from KoPrivateGPT.pipeline.base import BasePipeline
from KoPrivateGPT.pipeline.selector import ModuleSelector
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.schema import PipelineConfigAlias
from KoPrivateGPT.utils import slice_stop_words
from KoPrivateGPT.utils.embed import Embedding


class BasicIngestPipeline(BasePipeline):
    def __init__(self, file_loader_type: PipelineConfigAlias = ("file_loader", {"target_dir": Options.source_dir}),
                 text_splitter_type: PipelineConfigAlias = ("recursive_text_splitter", {"chunk_size": 500,
                                                                                        "chunk_overlap": 50}),
                 db_type: PipelineConfigAlias = ("mongo_db", {"mongo_url": MongoDBOptions.mongo_url,
                                                              "db_name": MongoDBOptions.db_name,
                                                              "collection_name": MongoDBOptions.collection_name}),
                 retrieval_type: PipelineConfigAlias = ("vector_db", {"vectordb_type": "chroma",
                                                                      "embedding": Embedding(embed_type="openai",
                                                                                             device_type="cuda")})):
        self.file_loader_type = file_loader_type
        self.text_splitter_type = text_splitter_type
        self.db_type = db_type
        self.retrieval_type = retrieval_type
        load_dotenv(verbose=False)

    def run(self, target_dir=None, *args, **kwargs):
        # File Loader
        if target_dir is not None:
            file_loader = ModuleSelector("file_loader").select(self.file_loader_type[0]).get(target_dir=target_dir)
        else:
            file_loader = ModuleSelector("file_loader").select(self.file_loader_type[0]).get(**self.file_loader_type[1])
        documents = file_loader.load()
        if len(documents) <= 0:
            return

        # Text Splitter
        splitter = ModuleSelector("text_splitter").select(self.text_splitter_type[0]).get(**self.text_splitter_type[1])
        passages = []
        for document in documents:
            passages.extend(splitter.split_document(document))
        print(f"Split into {len(passages)} passages")

        # Save passages to DB
        db = ModuleSelector("db").select(self.db_type[0]).get(**self.db_type[1])
        db.create_or_load()
        db.save(passages)

        # Ingest to retrieval
        retrieval_step = ModuleSelector("retrieval").select(self.retrieval_type[0])
        retrieval = retrieval_step.get(**self.retrieval_type[1])
        retrieval.ingest(passages)
        print("Ingest complete!")


class BasicDatasetPipeline(BasePipeline):
    def __init__(self, file_loader_type: PipelineConfigAlias, retrieval_type: PipelineConfigAlias):
        self.file_loader_type = file_loader_type
        self.retrieval_type = retrieval_type
        load_dotenv(verbose=False)

    def run(self, *args, **kwargs):
        # File Loader
        file_loader = ModuleSelector("file_loader").select(self.file_loader_type[0]).get(**self.file_loader_type[1])
        documents = file_loader.load()
        if len(documents) <= 0:
            return

        passages = [Passage(id=document.metadata['id'], content=document.page_content,
                            filepath='KoStrategyQA', previous_passage_id=None, next_passage_id=None) for document in
                    documents]
        # Ingest to retrieval
        retrieval = ModuleSelector("retrieval").select(self.retrieval_type[0]).get(**self.retrieval_type[1])
        retrieval.ingest(passages)


class BasicRunPipeline(BasePipeline):
    def __init__(self, db_type: PipelineConfigAlias = ("mongo_db", {"mongo_url": MongoDBOptions.mongo_url,
                                                                    "db_name": MongoDBOptions.db_name,
                                                                    "collection_name": MongoDBOptions.collection_name}),
                 retrieval_type: PipelineConfigAlias = ("vector_db", {"vectordb_type": "chroma",
                                                                      "embedding": Embedding(embed_type="openai",
                                                                                             device_type="cuda")}),
                 llm_type: PipelineConfigAlias = ("basic_llm", {"model_name": "gpt-3.5-turbo",
                                                                "api_base": None})):
        load_dotenv()
        self.db_type = db_type
        self.retrieval_type = retrieval_type
        self.llm_type = llm_type

    def run(self, query: str, *args, **kwargs) -> tuple[str, List[Passage]]:
        db = ModuleSelector("db").select(self.db_type[0]).get(**self.db_type[1])
        db.load()
        retrieval_step = ModuleSelector("retrieval").select(self.retrieval_type[0])
        retrieval = retrieval_step.get(**self.retrieval_type[1])
        llm = ModuleSelector("llm").select(self.llm_type[0]).get(**self.llm_type[1], db=db, retrieval=retrieval)
        answer, passages = llm.ask(query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        return answer, passages
