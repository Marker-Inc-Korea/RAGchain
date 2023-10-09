from typing import Optional, List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from KoPrivateGPT.preprocess.text_splitter.base import BaseTextSplitter
from KoPrivateGPT.schema import Passage


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
