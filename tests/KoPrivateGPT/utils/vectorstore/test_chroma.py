import os
import shutil

import chromadb
import pytest

from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.vectorstore import ChromaSlim
from base import PASSAGES, root_dir


@pytest.fixture
def chroma_slim():
    chroma_path = os.path.join(root_dir, "resources", "test_chroma_slim")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    chroma = ChromaSlim(
        client=chromadb.PersistentClient(path=chroma_path),
        collection_name='test_chroma_slim',
        embedding_function=EmbeddingFactory('openai').get()
    )
    yield chroma
    # teardown
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def test_chroma_slim(chroma_slim):
    chroma_slim.add_passages(PASSAGES)
    top_k = 2
    retrieved_docs = chroma_slim.similarity_search(query='I want to surf on the ocean.', k=top_k)
    assert len(retrieved_docs) == top_k
    assert retrieved_docs[0].metadata['passage_id'] == 'id-2'
