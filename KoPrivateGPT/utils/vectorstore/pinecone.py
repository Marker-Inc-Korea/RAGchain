from typing import List, Any, Optional

from langchain.vectorstores import Pinecone

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.vectorstore.base import SlimVectorStore


class PineconeSlim(Pinecone, SlimVectorStore):
    def add_passages(self, passages: List[Passage],
                     namespace: Optional[str] = None,
                     batch_size: int = 32,
                     **kwargs: Any):
        if namespace is None:
            namespace = self._namespace
        # Embed and make metadatas
        docs = []
        for passage in passages:
            embedding = self._embedding_function(passage.content)
            docs.append((passage.id, embedding, {'passage_id': passage.id}))

        self._index.upsert(
            vectors=docs,
            namespace=namespace,
            batch_size=batch_size,
            **kwargs
        )
