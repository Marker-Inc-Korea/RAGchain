from typing import List

from KoPrivateGPT.DB.pickle_db import PickleDB
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.preprocess.loader import FileLoader, KoStrategyQALoader
from KoPrivateGPT.preprocess.text_splitter import RecursiveTextSplitter
from KoPrivateGPT.retrieval import BM25Retrieval, VectorDBRetrieval
from KoPrivateGPT.llm.rerank import RerankLLM


def _text_modifier(text: str) -> List[str]:
    """
    You have to separate each word with underbar '_'
    """
    result = [text, text.lower(), text.capitalize(), text.upper()]
    if "_" in text:
        text_list = text.split("_")
        result.append("-".join(text_list))
        result.append("_".join([text.capitalize() for text in text_list]))
        result.append("-".join([text.capitalize() for text in text_list]))
        result.append("".join(text_list))
        result.append("".join([text.capitalize() for text in text_list]))
        result.append("".join([text.upper() for text in text_list]))
    return result


class ModuleSelector:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def select(self, name: str):
        if self.module_name in _text_modifier("file_loader"):
            self.select_file_loader(name)
        elif self.module_name in _text_modifier("text_splitter"):
            self.select_text_splitter(name)
        elif self.module_name in _text_modifier("db"):
            self.select_db(name)
        elif self.module_name in _text_modifier("retrieval"):
            self.select_retrieval(name)
        elif self.module_name in _text_modifier("llm"):
            self.select_llm(name)
        else:
            raise ValueError(f"Invalid module name: {self.module_name}")
        return self

    def get(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def select_file_loader(self, name: str):
        if name in _text_modifier("file_loader"):
            self.module = FileLoader
        elif name in _text_modifier("ko_strategy_qa_loader"):
            self.module = KoStrategyQALoader
        else:
            raise ValueError(f"Invalid module name: {name}")

    def select_text_splitter(self, name: str):
        if name in _text_modifier("recursive_text_splitter"):
            self.module = RecursiveTextSplitter
        else:
            raise ValueError(f"Invalid module name: {name}")

    def select_db(self, name: str):
        if name in _text_modifier("pickle_db"):
            self.module = PickleDB
        else:
            raise ValueError(f"Invalid module name: {name}")

    def select_retrieval(self, name: str):
        if name in _text_modifier("bm25"):
            self.module = BM25Retrieval
        elif name in _text_modifier("vector_db"):
            self.module = VectorDBRetrieval
        else:
            raise ValueError(f"Invalid module name: {name}")

    def select_llm(self, name: str):
        if name in _text_modifier("basic_llm"):
            self.module = BasicLLM
        elif name in _text_modifier("rerank_llm"):
            self.module = RerankLLM
        else:
            raise ValueError(f"Invalid module name: {name}")
