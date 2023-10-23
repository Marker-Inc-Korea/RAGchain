import copy
from typing import Optional, List, Tuple
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class HTMLheader_splitter(BaseTextSplitter):
    def __init__(
            self,
            headers_to_split_on: Optional[Tuple[str, str]] = None,
            return_each_element: bool = True,
    ):
        """
        :param headers_to_split_on: A list of Tuples to split on. Default is [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"),]
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

        # url일때, doc으로 들어올때 -> 홍창기 구조에서는 file loader에서 무조건 doc으로 받자는 규칙이기 때문에 고려하지 않아도 됌
        # Split HTML based HTML header.

        # 나오면 split된 Document 형태로 나오며 metadata는 상위 header 기준으로 어떻게 잘렸는지가 나옴.

        document_copy = copy.deepcopy(document)
        split_documents = self.html_splitter.split_text(document.page_content)

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
