from typing import List

from KoPrivateGPT.reranker.base import BaseReranker
from KoPrivateGPT.schema import Passage
from .base import Query, Text
from .transformer import MonoT5


class MonoT5Reranker(BaseReranker):
    """
    Rerank the passages using MonoT5 model.
    The model will be downloaded from HuggingFace model hub.
    """
    def __init__(self,
                 model_name: str = 'castorini/monot5-3b-msmarco-10k',
                 use_amp: bool = False,
                 token_false=None,
                 token_true=None,
                 *args, **kwargs):
        self.reranker = MonoT5(pretrained_model_name_or_path=model_name, use_amp=use_amp, token_false=token_false,
                               token_true=token_true)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        input_query = Query(text=query)
        input_texts = [Text(text=passage.content,
                            metadata={
                                'id': passage.id,
                                'filepath': passage.filepath,
                                'previous_passage_id': passage.previous_passage_id,
                                'next_passage_id': passage.next_passage_id,
                                'metadata_etc': passage.metadata_etc
                            }) for passage in passages]
        reranked_texts: List[Text] = self.reranker.rerank(input_query, input_texts)
        result_passage = [
            Passage(id=text.metadata['id'],
                    content=text.text,
                    filepath=text.metadata['filepath'],
                    previous_passage_id=text.metadata['previous_passage_id'],
                    next_passage_id=text.metadata['next_passage_id'],
                    metadata_etc=text.metadata['metadata_etc']) for text in reranked_texts]
        return result_passage

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("MonoT5 does not support sliding window reranking")
