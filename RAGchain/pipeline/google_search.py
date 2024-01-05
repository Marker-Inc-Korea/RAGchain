from typing import List

from langchain.llms import BaseLLM
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.schema import RAGchainPromptTemplate, Passage, RetrievalResult
from RAGchain.utils.websearch import GoogleSearch


class GoogleSearchRunPipeline(BaseRunPipeline):
    def __init__(self,
                 llm: BaseLLM,
                 prompt: RAGchainPromptTemplate = None,
                 ):
        """
        Initializes an instance of the GoogleSearchRunPipeline class.
        :param llm: An instance of the Langchain LLM module used for generating answers.
        :param prompt: RAGchainPromptTemplate used for generating prompts based on passages and user query.
        """
        self.llm = llm
        self.prompt = prompt if prompt is not None else self.default_prompt
        self.search = GoogleSearch()
        super().__init__()

    def _make_runnable(self):
        self.run = self.search | RunnableLambda(
            RetrievalResult.to_prompt_input) | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str], top_k: int = 5) -> tuple[
        List[str], List[List[Passage]], List[List[float]]]:
        runnable = self.search | {
            "passages": RunnableLambda(lambda x: x.passages),
            "scores": RunnableLambda(lambda x: x.scores),
            "answer": RunnableLambda(RetrievalResult.to_prompt_input) | self.prompt | self.llm | StrOutputParser()
        }
        results = runnable.batch(questions, config={"configurable": {"web_search_options": {"num_results": top_k}}})
        answers, passages, rel_scores = zip(
            *[(result["answer"], result["passages"], result["scores"]) for result in results])
        return list(answers), list(passages), list(rel_scores)
