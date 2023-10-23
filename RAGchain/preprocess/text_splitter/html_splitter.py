import copy
from typing import Optional, List, Tuple
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class HTMLHeaderSplitter(BaseTextSplitter):
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

        self.html_splitter = HTMLHeaderTextSplitter(headers_to_split_on, return_each_element)

    def split_document(self, document: Document) -> List[Passage]:

        # Split List[Document] by HTML header.
        document_copy = copy.deepcopy(document)
        split_documents = self.html_splitter.split_text(document.page_content)

        test_return_each_element = split_documents
        print('break point')
        # Convert List[Document] to List[Passage]
        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = document_copy.metadata.pop('source')  # user doc's metadata value.

        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            metadata_etc = dict(split_document.metadata.copy(),
                                **document_copy.metadata, )  # metadata_etc = doc's metadata_etc + headers

            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        print(f"Split into {len(passages)} passages")

        return passages
