from typing import List, Optional, Union

from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.reranker.base import BaseReranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainChatPromptTemplate, RAGchainPromptTemplate, RetrievalResult


class RerankRunPipeline(BaseRunPipeline):
    """
    Rerank pipeline is for question answering with retrieved passages using reranker.
    Af first, retrieval module will retrieve retrieve_size passages for reranking.
    Then, reranker rerank passages and use use_passage_count passages for llm question.


    :example:
    >>> from RAGchain.pipeline.rerank import RerankRunPipeline
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from RAGchain.reranker import MonoT5Reranker
    >>> from langchain.llms.openai import OpenAI

    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> reranker = MonoT5Reranker()
    >>> llm = OpenAI()
    >>> pipeline = RerankRunPipeline(retrieval, reranker, llm)
    >>> answer, passages, rel_scores = pipeline.get_passages_and_run(["What is the purpose of this framework based on the document?"])
    >>> print(answer[0])

    """

    def __init__(self, retrieval: BaseRetrieval, reranker: BaseReranker, llm: BaseLanguageModel,
                 prompt: Optional[Union[RAGchainPromptTemplate, RAGchainChatPromptTemplate]] = None,
                 use_passage_count: int = 5):
        """
        Initializes an instance of the RerankRunPipeline class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param reranker: An instance of the Reranker module used for reranking passages.
        :param llm: An instance of the Langchain LLM module used for generating answers.
        :param use_passage_count: The number of passages to use for llm question after reranking. Default is 5.
        """
        self.retrieval = retrieval
        self.reranker = reranker
        self.llm = llm
        self.prompt = self._get_default_prompt(llm, prompt)
        self.use_passage_count = use_passage_count
        super().__init__()

    def _make_runnable(self):
        self.run = self.retrieval | self.reranker | RunnableLambda(
            lambda x: x.slice(
                end=self.use_passage_count).to_prompt_input()) | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str], top_k: int = 5) -> tuple[
        List[str], List[List[Passage]], List[List[float]]]:
        runnable = self.retrieval | self.reranker | RunnableLambda(lambda x: x.slice(end=self.use_passage_count)) | {
            "passages": RunnableLambda(lambda x: x.passages),
            "scores": RunnableLambda(lambda x: x.scores),
            "answers": RunnableLambda(RetrievalResult.to_prompt_input) | self.prompt | self.llm | StrOutputParser()
        }
        result = runnable.batch([(question, top_k) for question in questions])
        answers, passages, rel_scores = zip(
            *[(answer['answers'], answer['passages'], answer['scores']) for answer in result])
        return answers, passages, rel_scores
