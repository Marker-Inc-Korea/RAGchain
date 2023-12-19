from operator import itemgetter
from typing import List, Optional, Union

from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.reranker.base import BaseReranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainChatPromptTemplate, RAGchainPromptTemplate


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
                 retrieval_option: Optional[dict] = None,
                 use_passage_count: int = 5):
        """
        Initializes an instance of the RerankRunPipeline class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param reranker: An instance of the Reranker module used for reranking passages.
        :param llm: An instance of the Langchain LLM module used for generating answers.
        :param retrieval_option: An optional dictionary of options for the retrieval module. Default is top_k=100.
        :param use_passage_count: The number of passages to use for llm question after reranking. Default is 5.
        """
        self.retrieval = retrieval
        self.reranker = reranker
        self.llm = llm
        self.prompt = self._get_default_prompt(llm, prompt)
        self.retrieval_option = retrieval_option if retrieval_option is not None else {"top_k": 100}
        self.use_passage_count = use_passage_count
        super().__init__()

    def _make_runnable(self):
        self.run = {
                       "passages": itemgetter("question") | RunnableLambda(lambda question: Passage.make_prompts(
                           self.__get_passages(question)[0])),
                       "question": itemgetter("question"),
                   } | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        passages, rel_scores = map(list, zip(*[self.__get_passages(question) for question in questions]))
        runnable = {
                       "question": itemgetter("question"),
                       "passages": itemgetter("passages") | RunnableLambda(lambda x: Passage.make_prompts(x))
                   } | self.prompt | self.llm | StrOutputParser()
        answers = runnable.batch([{"question": question, "passages": passage_group} for question, passage_group in
                                  zip(questions, passages)])
        return answers, passages, rel_scores

    def __get_passages(self, question: str):
        passages = self.retrieval.retrieve(question, **self.retrieval_option)
        reranked_passages = self.reranker.rerank(question, passages)[:self.use_passage_count]
        return reranked_passages, [i / len(reranked_passages) for i in range(len(reranked_passages), 0, -1)]
