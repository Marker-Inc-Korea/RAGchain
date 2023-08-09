from typing import List

from langchain import PromptTemplate, LLMChain

from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.model import load_model


class BasicLLM(BaseLLM):
    def __init__(self, model_type: str, device_type: str = 'cuda'):
        self.model = load_model(model_type, device_type)

    def ask(self, query: str, passages: List[Passage]) -> str:
        llm_chain = self._make_llm_chain(self.model)
        contents = "\n\n".join([passage.content for passage in passages])
        answer = llm_chain.run(context=contents, question=query)
        return answer

    def _make_llm_chain(self, llm):
        prompt_template = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
                    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.

                    {context}

                    질문: {question}
                    한국어 답변:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain
