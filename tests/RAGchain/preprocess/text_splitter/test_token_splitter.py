import os
import pathlib

import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import TokenSplitter

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_path = os.path.join(root_dir, "resources", "sample_test_document.txt")

with open(file_path) as f:
    state_of_the_union = f.read()

TEST_DOCUMENT = Document(
    page_content=state_of_the_union,
    metadata={
        'source': 'test_source',
        'Data information': '맨까 새끼들 부들부들하구나',
        'What is it?': 'THis is token splitter'
    }
)

@pytest.fixture
def tiktoken():
    tiktoken = TokenSplitter(tokenizer_name='tiktoken', chunk_size=1000, chunk_overlap=0)
    yield tiktoken

@pytest.fixture
def spaCy():
    spaCy = TokenSplitter(tokenizer_name='spaCy', chunk_size=1000, chunk_overlap=0)
    yield spaCy

@pytest.fixture
def sentence_transformers():
    sentence_transformers = TokenSplitter(tokenizer_name='SentenceTransformers', chunk_overlap=0)
    yield sentence_transformers

@pytest.fixture
def NLTK():
    NLTK = TokenSplitter(tokenizer_name='NLTK', chunk_size=1000)
    yield NLTK

@pytest.fixture
def Hugging_Face():
    Hugging_Face = TokenSplitter(tokenizer_name='huggingFace', chunk_size=100, chunk_overlap=0)
    yield Hugging_Face


def test_token_splitter(tiktoken, spaCy, sentence_transformers, NLTK, Hugging_Face):
    tiktoken_passages = tiktoken.split_document(TEST_DOCUMENT)

    spaCy_passages = spaCy.split_document(TEST_DOCUMENT)

    SentenceTransformers_passages = sentence_transformers.split_document(TEST_DOCUMENT)

    NLTK_passages = NLTK.split_document(TEST_DOCUMENT)

    huggingface_passages = Hugging_Face.split_document(TEST_DOCUMENT)

    test_passages = [tiktoken_passages, spaCy_passages, SentenceTransformers_passages, NLTK_passages, huggingface_passages]


    for passage in test_passages:
        assert len(passage) > 1
        assert passage[0].next_passage_id == passage[1].id
        assert passage[1].previous_passage_id == passage[0].id
        assert passage[0].filepath == 'test_source'
        assert passage[0].filepath == passage[1].filepath
        assert passage[0].previous_passage_id is None
        assert passage[-1].next_passage_id is None

    # Check if TEST_DOCUMENT content put in passages.
    for passage in test_passages:
        if passage == SentenceTransformers_passages:
            assert passage[0].content[:10] in TEST_DOCUMENT.page_content.strip()[:10].lower()
            assert passage[0].content[:10] in TEST_DOCUMENT.page_content.strip()[:10].lower()
        else:
            assert passage[0].content[:10] in TEST_DOCUMENT.page_content.strip()[:10]
            assert passage[0].content[:10] in TEST_DOCUMENT.page_content.strip()[:10]
