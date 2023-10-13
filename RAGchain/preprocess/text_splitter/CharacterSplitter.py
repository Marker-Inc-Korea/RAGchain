from typing import List, Any
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class CharacterSplitter(BaseTextSplitter):
    def __init__(self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any):
        """
        1. How the text is split: by single character.
        2. How the chunk size is measured: by number of characters.

        :param separators: A single string to split on. Default is "\n\n"
        :param is_separator_regex: Default is False.

        Optional
        **kwargs
        :param chunk_size: Maximum size of chunks to return. Default is 400        -> Because of CharacterTextSplitter class inherit TextSplitter
        :param chunk_overlap: Overlap in characters between chunks. Default is 200
        :param length_function: Function that measures the length of given chunks. Default len
        """

        """
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        """

        self.splitter = CharacterTextSplitter(separator, is_separator_regex)

    def split_document(self, document: Document) -> List[Passage]:
        """
        Convert Document to string.
        Split a document.
        """
        document_test = document
        document_test_type = type(document_test)
        split_documents = self.splitter.split_text(document)
        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]
        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            metadata_etc = split_document.metadata.copy()
            filepath = metadata_etc.pop('source')
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
        text_splitter.split_text(state_of_the_union)[0]
