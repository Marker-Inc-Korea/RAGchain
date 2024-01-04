from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage, RetrievalResult
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

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        input_query = Query(text=input.query)
        input_texts = list(map(self.__passage_to_text, input.passages))
        reranked_texts: List[Text] = self.reranker.rerank(input_query, input_texts)
        scores = list(map(lambda x: x.score, reranked_texts))
        result_passage = list(map(self.__text_to_passage, reranked_texts))
        input.passages = result_passage
        input.scores = scores
        return input

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        retrieval_result = RetrievalResult(query=query, passages=passages, scores=[])
        result = self.invoke(retrieval_result)
        return result.passages

    @staticmethod
    def __passage_to_text(passage: Passage) -> Text:
        return Text(text=passage.content,
                    metadata={
                        'id': passage.id,
                        'filepath': passage.filepath,
                        'content_datetime': passage.content_datetime,
                        'importance': passage.importance,
                        'previous_passage_id': passage.previous_passage_id,
                        'next_passage_id': passage.next_passage_id,
                        'metadata_etc': passage.metadata_etc
                    })

    @staticmethod
    def __text_to_passage(text: Text) -> Passage:
        return Passage(
            id=text.metadata['id'],
            content=text.text,
            filepath=text.metadata['filepath'],
            content_datetime=text.metadata['content_datetime'],
            importance=text.metadata['importance'],
            previous_passage_id=text.metadata['previous_passage_id'],
            next_passage_id=text.metadata['next_passage_id'],
            metadata_etc=text.metadata['metadata_etc']
        )
