from typing import List, Callable

from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base


class BasicLLM(BaseLLM):
    def __init__(self, retrieval: BaseRetrieval, model_name: str = "gpt-3.5-turbo", api_base: str = None,
                 prompt_func: Callable[[str, str], List[dict]] = None,
                 stream_func: Callable[[str], None] = None,
                 *args, **kwargs):
        super().__init__(retrieval)
        self.model_name = model_name
        set_api_base(api_base)
        self.get_message = self.get_messages if prompt_func is None else prompt_func
        self.stream_func = stream_func

    def ask(self, query: str, stream: bool = False, run_retrieve: bool = True, *args, **kwargs) -> tuple[
        str, List[Passage]]:
        passages = self.retrieved_passages if len(
            self.retrieved_passages) > 0 and not run_retrieve else self.retrieval.retrieve(query, top_k=4)
        contents = "\n\n".join([passage.content for passage in passages])
        answer = self.generate_chat(messages=self.get_message(contents, query),
                                    model=self.model_name,
                                    stream=stream,
                                    stream_func=self.stream_func,
                                    *args, **kwargs)
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
