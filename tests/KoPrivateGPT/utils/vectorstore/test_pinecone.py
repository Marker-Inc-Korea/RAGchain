import os

import pinecone
import pytest
from dotenv import load_dotenv

from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.vectorstore import PineconeSlim
from base import PASSAGES


@pytest.fixture(scope="session")
def pinecone_slim():
    load_dotenv()
    assert bool(os.getenv('PINECONE_API_KEY'))
    assert bool(os.getenv('PINECONE_ENV'))
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    index_name = "test-pinecone-slim"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
    index = pinecone.Index(index_name)
    pinecone_instance = PineconeSlim(
        index=index,
        embedding_function=EmbeddingFactory('openai').get().embed_query,
        text_key="text",
        namespace="test_pinecone_slim"
    )
    yield pinecone_instance
    # teardown
    index.delete(delete_all=True, namespace="test_pinecone_slim")


def test_pinecone_slim(pinecone_slim):
    pinecone_slim.add_passages(PASSAGES)
    top_k = 2
    retrieved_docs = pinecone_slim.similarity_search(query='I want to surf on the ocean.', k=top_k)
    assert len(retrieved_docs) == top_k
    assert retrieved_docs[0].metadata['passage_id'] == 'id-2'
