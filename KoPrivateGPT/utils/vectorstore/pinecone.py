from typing import List, Any, Optional

from langchain.vectorstores import Pinecone

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.vectorstore.base import SlimVectorStore


class PineconeSlim(Pinecone, SlimVectorStore):
    """
    Pinecone vector store stores only passage_id and vector.
    """
    def add_passages(self, passages: List[Passage],
                     namespace: Optional[str] = None,
                     batch_size: int = 32,
                     **kwargs: Any):
        if namespace is None:
            namespace = self._namespace
        # Embed and make metadatas
        vectors = []
        for passage in passages:
            embedding = self._embedding.embed_query(passage.content)
            vectors.append({
                'id': str(passage.id),
                'values': embedding,
                'metadata': {'passage_id': str(passage.id),
                             self._text_key: ""}
            })

        self._index.upsert(
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size,
            **kwargs
        )
