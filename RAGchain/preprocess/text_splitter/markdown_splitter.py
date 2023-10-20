import copy
from typing import List, Optional
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class MarkDownHeaderSplitter(BaseTextSplitter):
    def __init__(self, headers_to_split_on: Optional[List[tuple[str, str]]] = None, return_each_line: bool = False):
        """
        :param headers_to_split_on: A list of tuples which appended  to create split standard.
        ex)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        """

        # Set default value headers_to_split_on.
        if headers_to_split_on == None:
            headers_to_split_on = [("#", "Header 1")]

        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line)

    def split_document(self, documents: Document):
        document_copy = copy.deepcopy(documents)
        split_documents = self.markdown_splitter.split_text(documents.page_content)


        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = document_copy.metadata.pop('source')  # user doc's metadata value.



        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            metadata_etc = dict(split_document.metadata.copy(),
                                **documents.metadata, )  # metadata_etc = doc's metadata_etc + headers

            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)

            # Check splitter preserve other metadata in original document.
            assert passage.filepath in documents.metadata['source']
            # Check header value store into metadata_etc properly
            assert list(split_document.metadata.items())[0] == list(passage.metadata_etc.items())[0]

            passages.append(passage)
        print(f"Split into {len(passages)} passages")

        return passages

