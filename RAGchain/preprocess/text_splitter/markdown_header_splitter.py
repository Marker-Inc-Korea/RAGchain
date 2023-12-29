import copy
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter


class MarkDownHeaderSplitter(BaseTextSplitter):
    """
    The `MarkDownHeaderSplitter` is used to split a document into passages based document's header information which a list of separators contain.
    The most feature is similar with Langchain's MarkdownHeaderTextSplitter. It split based on header.
    """
    def __init__(self, headers_to_split_on: Optional[List[tuple[str, str]]] = None, return_each_line: bool = False):
        """
        :param headers_to_split_on: A list of tuples which appended to create split standard.
        ex)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        """

        # Set default value headers_to_split_on.
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]

        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line)

    def split_document(self, document: Document):
        doc_copy = copy.deepcopy(document)
        split_documents = self.markdown_splitter.split_text(document.page_content)
        for doc in split_documents:
            doc.metadata.update(doc_copy.metadata)
        passages = self.docs_to_passages(split_documents)
        return passages
