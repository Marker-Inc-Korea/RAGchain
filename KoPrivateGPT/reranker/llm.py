"""
The original code is from [RankGPT](https://github.com/sunnweiwei/RankGPT).
I modified the code to fit the KoPrivateGPT framework.
"""

from typing import List
import copy
from langchain.llms import BaseLLM

from KoPrivateGPT.reranker.base import BaseReranker
from KoPrivateGPT.schema import Passage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.chains import LLMChain


class LLMReranker(BaseReranker):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        messages = self.make_prompt(passages)
        chain = LLMChain(llm=self.llm, prompt=messages)
        # TODO: check if the input tokens are exceeded the limit of the LLM model : Feature/#105
        response = chain.run(query=query)
        print(response)
        return list()

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        pass

    def get_prefix_prompt(self, num):
        return [SystemMessagePromptTemplate.from_template("You are RankGPT, an intelligent assistant that can rank "
                                                          "passages based on their relevancy to the query."),
                HumanMessagePromptTemplate.from_template(f"I will provide you with {num} passages, each indicated by "
                                                         f"number identifier []. \nRank the passages based on their "
                                                         "relevance to query: {query}."),
                AIMessagePromptTemplate.from_template("Okay, please provide the passages."),
                ]

    def get_post_prompt(self, num):
        return HumanMessagePromptTemplate.from_template(
            "Search Query: {query}. \n" + f"Rank the {num} passages above based on their relevance to the search query. "
                                          "The passages should be listed in descending order using identifiers. The most relevant passages "
                                          "should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the "
                                          "ranking results, do not say any word or explain.")

    def make_prompt(self, passages: List[Passage]):
        num = len(passages)
        prompt_list = self.get_prefix_prompt(num)
        for i, passage in enumerate(passages):
            content = passage.content
            content = content.replace("\n", " ")
            content = content.replace('Title: Content: ', '')
            content = content.strip()
            prompt_list.append(HumanMessagePromptTemplate.from_template(f"[{i + 1}] {content}"))
            prompt_list.append(AIMessagePromptTemplate.from_template(f"Received passage [{i + 1}]."))
        prompt_list.append(self.get_post_prompt(num))
        return ChatPromptTemplate.from_messages(prompt_list)

    def clean_response(self, response: str):
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(self, response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(self, item, permutation, rank_start=0, rank_end=100):
        response = self.clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self.remove_duplicate(response)
        cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
            if 'rank' in item['hits'][j + rank_start]:
                item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
            if 'score' in item['hits'][j + rank_start]:
                item['hits'][j + rank_start]['score'] = cut_range[j]['score']
        return item
