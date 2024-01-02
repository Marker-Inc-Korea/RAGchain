from operator import itemgetter
from typing import List

from langchain.llms import BaseLLM
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.schema import RAGchainPromptTemplate, Passage
from RAGchain.utils.websearch import GoogleSearch


class GoogleSearchRunPipeline(BaseRunPipeline):
    def __init__(self,
                 llm: BaseLLM,
                 prompt: RAGchainPromptTemplate = None,
                 use_search_count: int = 3,
                 ):
        """
        Initializes an instance of the GoogleSearchRunPipeline class.
        :param llm: An instance of the Langchain LLM module used for generating answers.
        :param prompt: RAGchainPromptTemplate used for generating prompts based on passages and user query.
        :param use_search_count: The number of Google search result to be used for llm question answering. Default is 3.
        """
        self.llm = llm
        self.prompt = prompt if prompt is not None else self.default_prompt
        self.use_search_count = use_search_count
        super().__init__()

    def _make_runnable(self):
        self.run = {
                       "passages": itemgetter("question") | RunnableLambda(lambda question: Passage.make_prompts(
                           self.__search_passages(question)[0])),
                       "question": itemgetter("question"),
                   } | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        passages, rel_scores = map(list, zip(*[self.__search_passages(question) for question in questions]))
        runnable = {
                       "question": itemgetter("question"),
                       "passages": itemgetter("passages") | RunnableLambda(lambda x: Passage.make_prompts(x))
                   } | self.prompt | self.llm | StrOutputParser()
        answers = runnable.batch([{"question": question, "passages": passage_group} for question, passage_group in
                                  zip(questions, passages)])
        return answers, passages, rel_scores

    def __search_passages(self, query: str):
        search = GoogleSearch()
        passages = search.get_search_data(query, num_results=self.use_search_count)
        return passages, [i / len(passages) for i in range(len(passages), 0, -1)]
