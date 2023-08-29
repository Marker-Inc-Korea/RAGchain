from typing import List

import openai

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base


class BasicLLM(BaseLLM):
    def __init__(self, retrieval: BaseRetrieval, db: BaseDB, model_name: str = "gpt-3.5-turbo", api_base: str = None,
                 *args, **kwargs):
        self.retrieval = retrieval
        self.db = db
        self.model_name = model_name
        set_api_base(api_base)

    def ask(self, query: str) -> tuple[str, List[Passage]]:
        passages = self.retrieval.retrieve(query, self.db, top_k=4)
        contents = "\n\n".join([passage.content for passage in passages])
        completion = openai.ChatCompletion.create(model=self.model_name, messages=self.get_messages(contents, query),
                                                  temperature=0.5)
        answer = completion["choices"][0]["message"]["content"]
        return answer, passages

    @staticmethod
    def get_messages(context: str, question: str) -> List[dict]:
        system_prompt = f"""주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
                    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요."""
        user_prompt = f"""정보 : 
            {context}

            질문: {question}
            """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "다음은 질문에 대한 한국어 답변입니다. "}
        ]
