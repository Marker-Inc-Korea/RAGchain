import copy
from typing import Optional, List, Tuple

from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class HTMLHeaderSplitter(BaseTextSplitter):
    """
    The HTMLHeaderSplitter class in the RAGchain library is a text splitter that splits documents based on HTML headers.
    This class inherits from the BaseTextSplitter class and uses the HTMLHeaderTextSplitter.
    """
    def __init__(
            self,
            headers_to_split_on: Optional[Tuple[str, str]] = None,
            return_each_element: bool = False,
    ):
        """
        :param headers_to_split_on: list of tuples of headers we want to track mapped to (arbitrary) keys for metadata.
                                    Allowed header values: h1, h2, h3, h4, h5, h6
                                    Default is [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"),]
                                    e.g. [(“h1”, “Header 1”), (“h2”, “Header 2)].
        :param return_each_element: Return each element with associated headers. Default is False.
        """

        # Set headers_to_split_on default variable.
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3")
            ]

        self.html_header_splitter = HTMLHeaderTextSplitter(headers_to_split_on, return_each_element)

    def split_document(self, document: Document) -> List[Passage]:
        doc_copy = copy.deepcopy(document)
        split_documents = self.html_header_splitter.split_text(document.page_content)
        for doc in split_documents:
            doc.metadata.update(doc_copy.metadata)
        passages = self.docs_to_passages(split_documents)
        return passages
