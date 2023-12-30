from typing import Optional, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class RecursiveTextSplitter(BaseTextSplitter):
    """
    Split a document into passages by recursively splitting on a list of separators.
    You can specify a window_size and overlap_size to split the document into overlapping passages.
    """
    def __init__(self, separators: Optional[List[str]] = None,
                 keep_separator: bool = True,
                 *args, **kwargs):
        """
        :param separators: A list of strings to split on. Default is None.
        :param keep_separator: Whether to keep the separator in the passage. Default is True.
        :param kwargs: Additional arguments to pass to the langchain RecursiveCharacterTextSplitter.
        """
        self.splitter = RecursiveCharacterTextSplitter(separators, keep_separator, **kwargs)

    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document.
        """
        split_documents = self.splitter.split_documents([document])
        passages = Passage.from_documents(split_documents)
        return passages
