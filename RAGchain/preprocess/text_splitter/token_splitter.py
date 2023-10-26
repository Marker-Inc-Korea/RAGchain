import copy

from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, SpacyTextSplitter, SentenceTransformersTokenTextSplitter, NLTKTextSplitter
from transformers import GPT2TokenizerFast
from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class Token_Splitter(BaseTextSplitter):
    """
    Split a document into passages by recursively splitting on a list of separators.
    You can specify a window_size and overlap_size to split the document into overlapping passages.
    """
    def __init__(self, tokenizer_name: str = 'tiktoken', chunk_size: int = 0, chunk_overlap: int = 0, **kwargs):
        """
        :param tokenizer_name: A tokenizer_name name. You can choose tokenizer_name.
                        (tiktoken, spaCy, SentenceTransformers, NLTK, huggingFace)
        :param separators: A list of strings to split on. Default is None.
        :param keep_separator: Whether to keep the separator in the passage. Default is True.
        :param kwargs: Additional arguments to pass to the langchain RecursiveCharacterTextSplitter.
        """
        self.chosen_tokenizer = tokenizer_name

        # tiktoken (Default: chunk_size = 100, chunk_overlap = 0)
        self.tiktoken_splitter = TokenTextSplitter.from_tiktoken_encoder()

        # spaCy (Default: chunk_size = 10, chunk_overlap = 0)
        self.spaCy_splitter = SpacyTextSplitter()

        # SentenceTransformers (Default: chunk_overlap=0)
        self.SentenceTransformers_splitter = SentenceTransformersTokenTextSplitter()

        # NLTK
        self.NLTK_splitter = NLTKTextSplitter()

        # Hugging Face
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.huggingFace_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=100, chunk_overlap=0
        )



    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document.
        """
        global split_texts, split_documents


        doc_copy = copy.deepcopy(document)

        if self.chosen_tokenizer == 'tiktoken':
            split_texts = self.tiktoken_splitter.split_text(document.page_content)
            split_documents = self.tiktoken_splitter.create_documents(split_texts)

        elif self.chosen_tokenizer == 'spaCy':
            split_texts = self.spaCy_splitter.split_text(document.page_content)
            split_documents = self.spaCy_splitter.create_documents(split_texts)

        elif self.chosen_tokenizer == 'SentenceTransformers':
            split_texts = self.SentenceTransformers_splitter.split_text(document.page_content)
            split_documents = self.SentenceTransformers_splitter.create_documents(split_texts)

        elif self.chosen_tokenizer == 'NLTK':
            split_texts = self.NLTK_splitter.split_text(document.page_content)
            split_documents = self.NLTK_splitter.create_documents(split_texts)

        elif self.chosen_tokenizer == 'huggingFace':
            split_texts = self.huggingFace_splitter.split_text(document.page_content)
            split_documents = self.huggingFace_splitter.create_documents(split_texts)


        t = split_texts
        s = split_documents

        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = doc_copy.metadata.pop('source')  # user doc's metadata value.
        doc_metadata_etc = doc_copy.metadata  # TEST_DOCUMENT's metadata etc.(Already removed file path data)

        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            ## metadata_etc = doc's metadata_etc + headers
            metadata_etc = dict(split_document.metadata.copy(),  # Header information
                                **doc_metadata_etc, )  # TEST_DOCUMENT's metadata etc

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

