from typing import List

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class CodeSplitter(BaseTextSplitter):
    """
    The CodeSplitter class in the RAGchain library is a text splitter that splits documents
    based on separators of langchain's library Language enum. This class inherits from the
    BaseTextSplitter class and uses the from_language method of RecursiveCharacterTextSplitter
    class from the langchain library to perform the splitting.
    CodeSplitter supports CPP, GO, JAVA, KOTLIN, JS, TS, PHP, PROTO, `PYTHON`, RST, RUBY, RUST,
    SCALA, SWIFT, MARKDOWN, LATEX, HTML, SOL, CSHARP.
    """
    def __init__(
            self,
            language_name: str = 'PYTHON',
            chunk_size: int = 50,
            chunk_overlap: int = 0,
            **kwargs
    ):
        """
        :param language_name: A kind of language to split. Default is PYTHON.
        (CPP, GO, JAVA, KOTLIN, JS, TS, PHP, PROTO, PYTHON, RST, RUBY, RUST, SCALA, SWIFT, MARKDOWN, LATEX, HTML, SOL, CSHARP)
        :param chunk_size: Maximum size of chunks to return. Default is 50.
        :param chunk_overlap: Overlap in characters between chunks. Default is 0.
        :param kwargs: Additional arguments to pass to the langchain RecursiveCharacterTextSplitter.
        """

        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language[language_name], chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )

    def split_document(self, document: Document) -> List[Passage]:
        split_documents = self.code_splitter.split_documents([document])
        passages = self.docs_to_passages(split_documents)
        return passages
