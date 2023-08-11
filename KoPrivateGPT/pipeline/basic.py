from dotenv import load_dotenv
from typing import Dict, Any, List

from KoPrivateGPT.options import Options, DBOptions
from KoPrivateGPT.pipeline.base import BasePipeline
from KoPrivateGPT.pipeline.selector import ModuleSelector
from KoPrivateGPT.retrieval import BM25Retrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils import slice_stop_words
from KoPrivateGPT.utils.embed import Embedding


class BasicIngestPipeline(BasePipeline):
    file_loader_type: tuple[str, Dict[str, Any]] = "file_loader", {"target_dir": Options.source_dir},
    text_splitter_type: tuple[str, Dict[str, Any]] = "recursive_text_splitter", {"chunk_size": 500,
                                                                                 "chunk_overlap": 50},
    db_type: tuple[str, Dict[str, Any]] = "pickle_db", {"save_path": DBOptions.save_path},
    retrieval_type: tuple[str, Dict[str, Any]] = "vector_db", {"vectordb_type": "chroma",
                                                               "embedding_type": Embedding(embed_type="openai",
                                                                                           device_type="cuda"),
                                                               "device_type": "mps"}

    def run(self, *args, **kwargs):
        load_dotenv(verbose=False)
        # File Loader
        print(f"Loading documents from {Options.source_dir}")
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
        if retrieval_step.module is BM25Retrieval:
            self.retrieval_type[1]['db'] = db
        retrieval = retrieval_step.get(**self.retrieval_type[1])
        retrieval.ingest(passages)
        print("Ingest complete!")


def print_query_answer(query, answer):
    # Print the result
    print("\n\n> 질문:")
    print(query)
    print("\n> 대답:")
    print(answer)


def print_docs(docs: List[Passage]):
    # Print the relevant sources used for the answer
    print("----------------------------------참조한 문서---------------------------")
    for document in docs:
        print("\n> " + document.filepath + ":")
        print(document.content)
    print("----------------------------------참조한 문서---------------------------")


class BasicRunPipeline(BasePipeline):
    db_type: tuple[str, Dict[str, Any]] = "pickle_db", {"save_path": DBOptions.save_path},
    retrieval_type: tuple[str, Dict[str, Any]] = "vector_db", {"vectordb_type": "chroma",
                                                               "embedding_type": Embedding(embed_type="openai",
                                                                                           device_type="cuda"),
                                                               "device_type": "mps"}
    llm_type: tuple[str, Dict[str, Any]] = "basic_llm", {"device_type": "mps",
                                                         "model_type": "openai"}

    def run(self, *args, **kwargs):
        load_dotenv()

        db = ModuleSelector("db").select(self.db_type[0]).get(**self.db_type[1])
        db.load()

        retrieval_step = ModuleSelector("retrieval").select(self.retrieval_type[0])
        if retrieval_step.module is BM25Retrieval:
            self.retrieval_type[1]['db'] = db
        retrieval = retrieval_step.get(**self.retrieval_type[1])

        self.llm_type[1]['retrieval'] = retrieval
        llm = ModuleSelector("llm").select(self.llm_type[0]).get(**self.llm_type[1])

        while True:
            query = input("질문을 입력하세요: ")
            if query in ["exit", "종료"]:
                break

            answer, passages = llm.ask(query)
            answer = slice_stop_words(answer, ["Question :", "question:"])
            print_query_answer(query, answer)
            print_docs(passages)
